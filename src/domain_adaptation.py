import argparse

from src.config import (
    CHECKPOINTS_DIR, DOMAIN_ADAPT_LEARNING_RATE, DOMAIN_ADAPT_MAX_EPOCHS,
    DOMAIN_ADAPT_PATIENCE, EXPERIMENTS_DIR, MANIFEST_JOINT_TRAIN,
    MANIFEST_JOINT_VAL, MODEL_NAMES, SEED,
)
from src.evaluate import evaluate_model
from src.generalization import run_ood_evaluation
from src.train import train_model
from src.utils import save_json, seed_everything


def adapt_model(model_name: str) -> dict:
    source_checkpoint = CHECKPOINTS_DIR / f"{model_name}.pt"
    if not source_checkpoint.exists():
        raise FileNotFoundError(
            f"No PlantVillage checkpoint for {model_name} at {source_checkpoint} — "
            "train the base models first."
        )

    result = train_model(
        model_name,
        manifest_train=MANIFEST_JOINT_TRAIN,
        manifest_val=MANIFEST_JOINT_VAL,
        max_epochs=DOMAIN_ADAPT_MAX_EPOCHS,
        patience=DOMAIN_ADAPT_PATIENCE,
        tag="da",
        learning_rate=DOMAIN_ADAPT_LEARNING_RATE,
        init_checkpoint=source_checkpoint,
    )
    return result


def run_domain_adaptation(model_names=MODEL_NAMES) -> dict:
    seed_everything(SEED)
    for model_name in model_names:
        print(f"=== Domain-adapting {model_name} on PlantDoc ===")
        adapt_model(model_name)

    zero_shot = run_ood_evaluation(model_names)
    adapted = run_ood_evaluation(model_names, checkpoint_tag="da")

    comparison = {}
    for model_name in model_names:
        z = zero_shot.get(model_name, {})
        a = adapted.get(model_name, {})

        pv_after_adapt = evaluate_model(model_name, tag="da")["accuracy"]
        pv_before_adapt = z.get("in_distribution_accuracy")

        comparison[model_name] = {
            "zero_shot_ood_accuracy": z.get("ood_accuracy"),
            "adapted_ood_accuracy": a.get("ood_accuracy"),
            "improvement": (
                a["ood_accuracy"] - z["ood_accuracy"]
                if z.get("ood_accuracy") is not None and a.get("ood_accuracy") is not None
                else None
            ),
            "plantvillage_accuracy_before_adapt": pv_before_adapt,
            "plantvillage_accuracy_after_adapt": pv_after_adapt,
        }
        print(f"[Domain adaptation] {model_name}: OOD zero-shot={z.get('ood_accuracy')} "
              f"-> adapted={a.get('ood_accuracy')}; PlantVillage {pv_before_adapt} -> {pv_after_adapt}")

    save_json(comparison, EXPERIMENTS_DIR / "domain_adaptation_results.json")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="adapt a single model instead of all")
    args = parser.parse_args()
    if args.model:
        run_domain_adaptation([args.model])
    else:
        run_domain_adaptation()
