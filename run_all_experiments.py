import argparse

from src.config import IMAGES_PER_CLASS, MODEL_NAMES
from src.data.prepare_dataset import main as prepare_data
from src.evaluate import evaluate_model
from src.generalization import run_cross_validation, run_ood_evaluation, run_robustness_evaluation
from src.generate_report import main as generate_report
from src.run_xai import main as run_xai
from src.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny 1-epoch run to sanity-check the pipeline")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-generalization", action="store_true")
    parser.add_argument("--skip-xai", action="store_true")
    args = parser.parse_args()

    if not args.skip_data:
        prepare_data(images_per_class=20 if args.smoke else IMAGES_PER_CLASS)

    epochs = 1 if args.smoke else None
    for model_name in MODEL_NAMES:
        print(f"\n=== Training {model_name} ===")
        if epochs:
            train_model(model_name, max_epochs=epochs)
        else:
            train_model(model_name)
        print(f"=== Evaluating {model_name} ===")
        evaluate_model(model_name)

    if not args.skip_generalization:
        cv_epochs = 2 if args.smoke else 8
        print("\n=== Generalization: cross-validation ===")
        run_cross_validation(MODEL_NAMES, epochs=cv_epochs)
        print("\n=== Generalization: OOD (PlantDoc) ===")
        run_ood_evaluation(MODEL_NAMES)
        print("\n=== Generalization: robustness ===")
        run_robustness_evaluation(MODEL_NAMES)

    if not args.skip_xai:
        print("\n=== XAI ===")
        run_xai()

    print("\n=== Generating report ===")
    generate_report()


if __name__ == "__main__":
    main()
