import torch
from os.path import join
from torch.utils.data import DataLoader
from sectors.config import DATA_DIR
from sectors.utils.models import set_seed
from sectors.experiments.prompt_tuning.model import construct_generation_args


args = construct_generation_args()
set_seed(args.seed)
trainer = args.TRAINER_CLASS(args)

TEST_PATH = join(DATA_DIR, args.dataset, "test_preprocessed.json")
trainer.test_set = trainer.get_dataset(TEST_PATH, trainer.args)
trainer.test_loader = DataLoader(trainer.test_set, batch_size=trainer.args.batch_size)

factor = 100
# inference flops without trie search
with torch.profiler.profile(with_flops=True) as inf_profiler:
    trainer.test_evaluation(0, trainer.test_loader, "Test", False, factor)
ifs = sum(event.flops for event in inf_profiler.key_averages())
inference_flops = (
    (ifs * factor) / len(trainer.test_loader)
) * 1e6
print("Inference Flops", inference_flops)

if args.head == "lh":
    # inference flops with trie search
    with torch.profiler.profile(with_flops=True) as inf_ts_profiler:
        trainer.test_evaluation(
            0, trainer.test_loader, "Test", True, factor
        )
    ifts = sum(event.flops for event in inf_ts_profiler.key_averages())
    inference_flops_trie_search = (
        (ifts * factor) / len(trainer.test_loader)
    ) * 1e6
    print("Inference Flops Trie Search", inference_flops_trie_search)