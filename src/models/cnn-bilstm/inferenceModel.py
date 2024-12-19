import cv2
import numpy as np
import os

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from ctc_beam_search_decoder import beam_search_decoder

BEAM_SEARCH = True
BEAM_WIDTH = 3
MODEL_NAME = "202412070826"
WANDB_ID = "k02j95c1"


def load_possible_moves(file_path: str) -> set:
    """
    Loads possible moves from the all_moves_proba.txt file and stores them in a set for quick lookup
    """
    possible_moves = set()
    with open(file_path, "r") as file:
        for line in file:
            move = line.split(",")[0].strip()
            possible_moves.add(move)
    return possible_moves

def is_move_in_list(move: str, possible_moves: set) -> bool:
    """
    Checks if a given move is present in list of possible moves
    """
    return move in possible_moves

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.metadata["vocab"])[0]

        return text

    def predict_top_k(self, image: np.ndarray, beam_width: int = 3):
        """
        Predicts the top k most probable sequences with confidence scores.

        Args:
            image (np.ndarray): Input image.
            beam_width (int): Number of top predictions to return.

        Returns:
            List[tuple]: List of (decoded_text, confidence_score) tuples.
        """
        # Preprocess the image
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        # Run the model
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        preds = preds[0]  # Remove batch dim, now preds is of shape (T=64, C=28), with T=number of timesteps, C=number of classes

        # Apply softmax if not already done
        if not np.allclose(preds.sum(axis=-1), 1.0):
            preds = np.exp(preds) / np.exp(preds).sum(axis=-1, keepdims=True)

        # Use beam search decoder
        top_predictions = beam_search_decoder(preds, self.metadata["vocab"], beam_width=beam_width)

        return top_predictions

    def predict_top_valid(self, image: np.ndarray, possible_moves: set, beam_width: int = 3):
        """
        Predicts the most probable valid sequence or the most confident invalid sequence.

        Args:
            image (np.ndarray): Input image.
            possible_moves (set): Set of valid moves.
            beam_width (int): Number of top predictions to evaluate.

        Returns:
            tuple: (predicted_text, confidence_score, is_valid)
        """
        # Preprocess the image
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        # Run the model
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        # Remove batch dimension
        preds = preds[0]

        # Apply softmax to get probabilities
        preds = np.exp(preds) / np.exp(preds).sum(axis=-1, keepdims=True)

        # Use beam search decoder
        top_predictions = beam_search_decoder(preds, self.metadata["vocab"], beam_width=beam_width)

        print(f'Top predictions: {top_predictions}')

        # Find the first valid prediction
        for predicted_text, confidence in top_predictions:
            if is_move_in_list(predicted_text, possible_moves):
                return predicted_text, confidence, True  # Return if valid

        # If no valid prediction is found, return the most confident one
        most_confident = top_predictions[0]
        return most_confident[0], most_confident[1], False  # Return the most confident (invalid)


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    import wandb

    print("Make sure to run this script from the project's root folder !")

    # wandb.init(project="handwriting_recognition", resume=True, id=WANDB_ID) # change ID !
    # wandb.init(project="handwriting_recognition")

    # Change model name to infer with !
    model_path = f"models/handwriting_recognition_torch/{MODEL_NAME}/best_model.onnx"
    model = ImageToWordModel(model_path=model_path)

    df = pd.read_csv("data/test_data/val.csv").values.tolist()

    possible_moves = load_possible_moves("/Users/vincentfiszbin/Projets/cs433_project_2/src/data_generation/all_moves_proba.txt")
    valid_moves = []
    invalid_moves = []

    accum_cer = []
    accum_wer = []
    error_cases = []
    preds_and_labels_padded = []

    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        if BEAM_SEARCH == False:
            predicted_text = model.predict(image)
            confidence = None
        else:
            predicted_text, confidence, is_valid = model.predict_top_valid(image, possible_moves, beam_width=BEAM_WIDTH)
        
        # Check move validity
        if is_move_in_list(predicted_text, possible_moves):
            valid_moves.append(predicted_text)
        else:
            invalid_moves.append(predicted_text)

        if type(label) == float: # Added to handle float type labels
            print(f"label has float type, converting to string")
            label = str(label)

        padded_label = label
        if len(padded_label) < 7:
            b = ['.'] * (7 - len(padded_label))
            padded_label += ''.join(b)

        padded_predicted_text = predicted_text
        if len(padded_predicted_text) < 7:
            b = ['.'] * (7 - len(padded_predicted_text))
            padded_predicted_text += ''.join(b)

        cer = get_cer(predicted_text, label)
        wer = get_wer(predicted_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {predicted_text}, Confidence: {confidence}, CER: {cer}, WER: {wer}\n")

        accum_cer.append(cer)
        accum_wer.append(wer)

        # Error collection
        if predicted_text != label:
            error_cases.append({
                "image_path": image_path,
                "label": label,
                "prediction": predicted_text,
                "confidence": confidence,
                "CER": cer,
                "WER": wer
            })
        preds_and_labels_padded.append(
            {
                "label": padded_label,
                "prediction": padded_predicted_text
            }
        )

    print(f"Total moves: {len(df)}")
    print(f"Valid moves: {len(valid_moves)}")
    print(f"Invalid moves: {len(invalid_moves)}")
    print("Invalid moves:", invalid_moves)

    avg_cer = np.average(accum_cer)
    avg_wer = np.average(accum_wer)
    total_accuracy = (1 - avg_wer) * 100

    print(f"Average CER: {avg_cer}")
    print(f"Average WER: {avg_wer}")
    print(f"Total Accuracy: {total_accuracy:.2f}%")

    # Save error cases
    model_dir = os.path.dirname(model_path)
    output_dir = os.path.join(model_dir, "error_analysis")
    os.makedirs(output_dir, exist_ok=True)
    for i, error in enumerate(error_cases):
        image = cv2.imread(error["image_path"])
        cv2.putText(image, f"Label: {error['label']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
        cv2.putText(image, f"Pred: {error['prediction']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, f"Conf: {error['confidence']:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        cv2.imwrite(os.path.join(output_dir, f"error_{i}.png"), image)
    error_df = pd.DataFrame(error_cases)
    error_df.to_csv(os.path.join(output_dir, "error_analysis.csv"), index=False)

    preds_and_labels_padded_df = pd.DataFrame(preds_and_labels_padded)
    preds_and_labels_padded_df.to_csv(os.path.join(output_dir, "preds_and_labels_padded.csv"), index=False)

    # #Log metrics to wandb
    # wandb.log({
    #     "final_test_average_CER": avg_cer,
    #     "final_test_average_WER": avg_wer,
    #     "final_test_total_accuracy": total_accuracy,
    # })
    # # Log errors to wandb
    # error_table = wandb.Table(columns=["Image", "Label", "Prediction", "Confidence", "CER", "WER"])
    # for error in error_cases:
    #     image = cv2.imread(error["image_path"])
    #     error_table.add_data(
    #         wandb.Image(image, caption=f"Label: {error['label']} | Prediction: {error['prediction']}"),
    #         error["label"],
    #         error["prediction"],
    #         error["confidence"],
    #         error["CER"],
    #         error["WER"]
    #     )
    # wandb.log({"Error Analysis Table": error_table})