# Setup Environment

```bash
conda env create -f env.yml
```

# Run MWE
```bash
cd mwe
# before submitting the job, set project allocation in run.sh
# do not modify the dataset and partition dirs
sbatch run.sh
```