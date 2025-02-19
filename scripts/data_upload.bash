python data_creation/dataset_creat.py \
  --paths \
    sciq \
    commonsense_qa \
    winogrande \
    openbookqa \
    reclor \
    arc_easy \
    logiqa \
    svamp \
    gsm8k \
  --repo_name llama \
  --test_ratio 0.1 \
  --model "llama" \
  --delete_local_files


