#!/bin/bash
##

## Eval Generation ------------------------------------------------------

##---------- DiT Eval Generation
#python inference/nitro_t_geneval_quant.py geneval/prompts/evaluation_metadata.jsonl --outdir outputs/nitro-t-0.6B-Wint8 --model models/Nitro-T-0.6B --resolution 512 |&tee log_Nitro-T-0.6B_Wint8.txt

## GenEval Evaluation -------------------------------------------------------

## Evaluation---
# python geneval/evaluation/evaluate_images.py outputs/nitro-t-0.6B-Wint8 --outfile outputs/results_nitro-t-06B-Wint8.jsonl --model-path geneval/Object_detector_folder

## Scoring---
# python geneval/evaluation/summary_scores.py outputs/results_nitro-t-06B-Wint8.jsonl


##---------- MMDiT Eval Generation
python inference/nitro_t_geneval_quant.py geneval/prompts/evaluation_metadata.jsonl --outdir outputs/nitro-t-1.2B-Wint8_nw --model models/Nitro-T-1.2B --resolution 1024 |&tee log_Nitro-T-1.2B_Wint8_16-02-2026.txt

# python geneval/evaluation/evaluate_images.py outputs/nitro-t-1.2B-Wint8_nw --outfile outputs/results_nitro-t-12B-Wint8_nw.jsonl --model-path geneval/Object_detector_folder

# python geneval/evaluation/summary_scores.py outputs/results_nitro-t-12B-Wint8_nw.jsonl