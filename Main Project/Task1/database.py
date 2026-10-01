import os
from datetime import datetime
import pandas as pd


class Database:

    def __init__(self, db_path: str = "DB/database.csv"):
        self.db_path = db_path
        self.columns = [
            "timestamp",
            "input_type",
            "user_text",
            "generated_caption",
            "combined_text",
            "predicted_label",
            "confidence",
            "model_used",
        ]
        self._init_db()

    def _init_db(self) -> None:
        # Ensure target directory exists
        dir_name = os.path.dirname(self.db_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # Initialize CSV file with headers if missing
        if not os.path.exists(self.db_path):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.db_path, index=False)

    def log_entry(
        self,
        input_type: str,
        user_text: str = "",
        generated_caption: str = "",
        combined_text: str = "",
        predicted_label: str = "",
        confidence: float = 0.0,
        model_used: str = "BiLSTM",
    ) -> None:
        """
        Parameters:
            input_type: 'Text Only', 'Image Only', or 'Multimodal (Text + Image)'
            user_text: Raw query entered by the user.
            generated_caption: Caption output from BLIP (if image uploaded).
            combined_text: Final merged string passed to the classifier.
            predicted_label: Output class from the classifier.
            confidence: Confidence score (0.0 to 1.0).
            model_used: Name/identifier of the model architecture used.
        """
        new_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_type": input_type,
            "user_text": user_text or "",
            "generated_caption": generated_caption or "",
            "combined_text": combined_text or "",
            "predicted_label": predicted_label,
            "confidence": round(float(confidence), 4),
            "model_used": model_used,
        }

        df_new = pd.DataFrame([new_record])
        df_new.to_csv(self.db_path, mode="a", header=False, index=False)

    def get_history(self) -> pd.DataFrame:
        
        if not os.path.exists(self.db_path):
            return pd.DataFrame(columns=self.columns)

        try:
            df = pd.read_csv(self.db_path)
            return df.iloc[::-1].reset_index(drop=True)# Display newest entries first
        except Exception:
            return pd.DataFrame(columns=self.columns)

    def clear_history(self) -> None:
        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.db_path, index=False)