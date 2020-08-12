"""
How torun : 
    CUDA_VISIBLE_DEVICES=0 python main_runner.py --output_dir ./tmp/ --do_train --do_eval --data_path ~/dataset/sessions_with_neg_samples_example/ --per_device_train_batch_size 128 --model_type xlnet
"""
import os
import sys
import math
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import Counter

import yaml
from transformers import (
    HfArgumentParser,
    set_seed,
)

from recsys_models import get_recsys_model
from recsys_meta_model import RecSysMetaModel
from recsys_trainer import RecSysTrainer
from recsys_metrics import EvalMetrics
from recsys_args import DataArguments, ModelArguments, TrainingArguments
from recsys_data import (
    fetch_data_loaders,
    get_avail_data_dates
)

logger = logging.getLogger(__name__)

# this code use Version 3
assert sys.version_info.major > 2

def main():

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    with open(data_args.feature_config) as yaml_file:
        feature_map = yaml.load(yaml_file, Loader=yaml.FullLoader)
    target_size = feature_map['sess_pid_seq']['cardinality']

    seq_model, config = get_recsys_model(model_args, data_args, training_args, target_size)
    rec_model = RecSysMetaModel(seq_model, config, model_args, data_args, feature_map)

    eval_metrics_all, eval_metrics_neg = EvalMetrics(ks=[5,10,100,1000]), EvalMetrics(ks=[5,10,50])

    trainer = RecSysTrainer(
        model=rec_model,
        args=training_args,
        compute_metrics_all=eval_metrics_all,
        compute_metrics_neg=eval_metrics_neg,
        fast_test=model_args.fast_test
    )

    trainer.update_wandb_args(model_args)
    trainer.update_wandb_args(data_args)

    data_dates = get_avail_data_dates(data_args)
    results_dates_all = {}
    results_dates_neg = {}

    for date_idx in range(1, len(data_dates)):
        train_date, eval_date, test_date = data_dates[date_idx - 1], data_dates[date_idx -1], data_dates[date_idx]

        train_loader, eval_loader, test_loader \
            = fetch_data_loaders(data_args, training_args, feature_map, train_date, eval_date, test_date)

        trainer.set_rec_train_dataloader(train_loader)
        trainer.set_rec_eval_dataloader(eval_loader)
        trainer.set_rec_test_dataloader(test_loader)

        # Training
        if training_args.do_train:
            logger.info("*** Train (date:{})***".format(train_date))

            model_path = (
                model_args.model_name_or_path
                if model_args.model_name_or_path is not None \
                    and os.path.isdir(model_args.model_name_or_path)
                else None
            )
            trainer.train(model_path=model_path)

        # Evaluation (on testset)
        if training_args.do_eval:
            logger.info("*** Evaluate (date:{})***".format(test_date))

            eval_output = trainer.predict()
            eval_metrics_all = eval_output.metrics_all
            eval_metrics_neg = eval_output.metrics_neg

            output_eval_file = os.path.join(training_args.output_dir, "eval_results_dates.txt")
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results (all) (date:{})*****".format(test_date))
                    writer.write("***** Eval results (all) (date:{})*****".format(test_date))
                    for key in sorted(eval_metrics_all.keys()):
                        logger.info("  %s = %s", key, str(eval_metrics_all[key]))
                        writer.write("%s = %s\n" % (key, str(eval_metrics_all[key])))
                    
                    logger.info("***** Eval results (neg) (date:{})*****".format(test_date))
                    writer.write("***** Eval results (neg) (date:{})*****".format(test_date))
                    for key in sorted(eval_metrics_neg.keys()):
                        logger.info("  %s = %s", key, str(eval_metrics_neg[key]))
                        writer.write("%s = %s\n" % (key, str(eval_metrics_neg[key])))

            results_dates_all[test_date] = eval_metrics_all
            results_dates_neg[test_date] = eval_metrics_neg
        
    logger.info("train and eval for all dates are done")
    trainer.save_model()

    if training_args.do_eval and trainer.is_world_master():
        
        eval_df_all = pd.DataFrame.from_dict(results_dates_all, orient='index')
        eval_df_neg = pd.DataFrame.from_dict(results_dates_neg, orient='index')
        np.save(os.path.join(training_args.output_dir, "eval_results_dates_all.npy"), eval_df_all)
        np.save(os.path.join(training_args.output_dir, "eval_results_dates_neg.npy"), eval_df_neg)
        eval_avg_days_all = dict(eval_df_all.mean())
        eval_avg_days_neg = dict(eval_df_neg.mean())
        output_eval_file = os.path.join(training_args.output_dir, "eval_results_avg.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results (all) (avg over dates)*****")
            for key in sorted(eval_avg_days_all.keys()):
                logger.info("  %s = %s", key, str(eval_avg_days_all[key]))
                writer.write("%s = %s\n" % (key, str(eval_avg_days_all[key])))
            trainer._log({f"AOD_all_{k}":v for k, v in eval_avg_days_all.items()})
            
            logger.info("***** Eval results (neg) (avg over dates)*****")
            for key in sorted(eval_avg_days_neg.keys()):
                logger.info("  %s = %s", key, str(eval_avg_days_neg[key]))
                writer.write("%s = %s\n" % (key, str(eval_avg_days_neg[key])))
            trainer._log({f"AOD_neg_{k}":v for k, v in eval_avg_days_neg.items()})
                
    return results_dates_all


if __name__ == "__main__":
    main()      