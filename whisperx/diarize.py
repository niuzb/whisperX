import numpy as np
import pandas as pd
from pyannote.audio import Pipeline, Model, Inference
from typing import Optional, Union, List, Dict
import torch

from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.schema import TranscriptionResult, AlignedTranscriptionResult
from whisperx.log_utils import get_logger

logger = get_logger(__name__)


class DiarizationPipeline:
    def __init__(
        self,
        model_name=None,
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        model_config = model_name or "pyannote/speaker-diarization-3.1"
        logger.info(f"Loading diarization model: {model_config}")
        self.model = Pipeline.from_pretrained(model_config, use_auth_token=use_auth_token).to(device)

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> Union[tuple[pd.DataFrame, Optional[dict[str, list[float]]]], pd.DataFrame]:
        """
        Perform speaker diarization on audio.
        """
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }

        if return_embeddings:
            diarization, embeddings = self.model(
                audio_data,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=True,
            )
        else:
            diarization = self.model(
                audio_data,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            embeddings = None

        diarize_df = pd.DataFrame(diarization.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)

        if return_embeddings and embeddings is not None:
            speaker_embeddings = {speaker: embeddings[s].tolist() for s, speaker in enumerate(diarization.labels())}
            return diarize_df, speaker_embeddings
        
        # For backwards compatibility
        if return_embeddings:
            return diarize_df, None
        else:
            return diarize_df


class SpeechEmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "pyannote/wespeaker-voxceleb-resnet34-LM",
        use_auth_token: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = "cpu"
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = Model.from_pretrained(model_name, use_auth_token=use_auth_token)
        self.model = self.model.to(device)
        self.inference = Inference(self.model, window="whole")

    def __call__(self, audio_segment: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract embedding from an audio segment.
        Args:
            audio_segment: numpy array or torch tensor of shape (1, num_samples) or (num_samples,)
        """
        if isinstance(audio_segment, np.ndarray):
            if audio_segment.ndim == 1:
                audio_segment = audio_segment[None, :]
            waveform = torch.from_numpy(audio_segment).float()
        else:
            if audio_segment.ndim == 1:
                waveform = audio_segment.unsqueeze(0)
            else:
                waveform = audio_segment
        
        waveform = waveform.to(self.device)
        # pyannote Inference expects tensor of shape (1, time)
        with torch.no_grad():
            embedding = self.inference({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        return embedding


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    speaker_embeddings: Optional[dict[str, list[float]]] = None,
    fill_nearest: bool = False,
) -> Union[AlignedTranscriptionResult, TranscriptionResult]:
    """
    Assign speakers to words and segments in the transcript.
    """
    transcript_segments = transcript_result["segments"]
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'], seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])
        # remove no hit, otherwise we look for closest (even negative intersection...)
        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        else:
            dia_tmp = diarize_df
        if len(dia_tmp) > 0:
            # sum over speakers
            speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            seg["speaker"] = speaker
        
        # assign speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'], word['start'])
                    # remove no hit
                    if not fill_nearest:
                        dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                    else:
                        dia_tmp = diarize_df
                    if len(dia_tmp) > 0:
                        # sum over speakers
                        speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                        word["speaker"] = speaker

    # Add speaker embeddings to the result if provided
    if speaker_embeddings is not None:
        transcript_result["speaker_embeddings"] = speaker_embeddings

    return transcript_result


class Segment:
    def __init__(self, start:int, end:int, speaker:Optional[str]=None):
        self.start = start
        self.end = end
        self.speaker = speaker
