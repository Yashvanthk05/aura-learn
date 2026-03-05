import { useCallback, useState, useRef } from "react";
import { Play, Pause, Upload, X } from "lucide-react";
import { useApp } from "../../store/AppContext";
import * as api from "../../api/client";

export default function AudioFeature() {
  const { state, dispatch } = useApp();
  const [playing, setPlaying] = useState(false);
  const [audioEl, setAudioEl] = useState(null);
  const [maxLength, setMaxLength] = useState(150);
  const [minLength, setMinLength] = useState(40);
  const [speakerAudio, setSpeakerAudio] = useState(null);
  const fileInputRef = useRef(null);

  const run = useCallback(async () => {
    dispatch({ type: "SET_FEATURE_LOADING", payload: true });
    try {
      const res = await api.summarizeAndAudio(state.activeDocumentId, {
        type: "abstractive",
        maxLength,
        minLength,
        speakerAudio,
      });
      dispatch({ type: "SET_FEATURE_RESULT", payload: res });
    } catch (err) {
      dispatch({ type: "SET_FEATURE_ERROR", payload: err.message });
    }
  }, [state.activeDocumentId, maxLength, minLength, speakerAudio, dispatch]);

  const togglePlay = useCallback(() => {
    if (!audioEl) return;
    if (playing) {
      audioEl.pause();
    } else {
      audioEl.play();
    }
    setPlaying(!playing);
  }, [audioEl, playing]);

  const handleFileChange = useCallback((e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSpeakerAudio(file);
    }
  }, []);

  const clearSpeakerAudio = useCallback(() => {
    setSpeakerAudio(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, []);

  return (
    <div className='space-y-4'>
      <div>
        <label
          className='text-xs mb-2 block font-medium'
          style={{ color: "var(--fg-secondary)" }}
        >
          Voice Clone (Optional)
        </label>
        <div className='flex items-center gap-2'>
          <input
            ref={fileInputRef}
            type='file'
            accept='audio/*'
            onChange={handleFileChange}
            className='hidden'
            id='speaker-audio-input'
          />
          <label
            htmlFor='speaker-audio-input'
            className='flex-1 cursor-pointer px-3 py-2 rounded-lg text-xs transition-colors flex items-center gap-2'
            style={{
              background: "var(--bg-elevated)",
              border: "1px solid var(--border)",
              color: "var(--fg-secondary)",
            }}
          >
            <Upload size={14} />
            {speakerAudio ? speakerAudio.name : "Upload voice sample"}
          </label>
          {speakerAudio && (
            <button
              onClick={clearSpeakerAudio}
              className='p-2 rounded-lg transition-colors'
              style={{
                background: "var(--bg-elevated)",
                border: "1px solid var(--border)",
                color: "var(--fg-secondary)",
              }}
              title='Clear audio file'
            >
              <X size={14} />
            </button>
          )}
        </div>
        <p className='text-xs mt-1' style={{ color: "var(--fg-muted)" }}>
          Upload a short audio clip (5-10 seconds) to clone this voice
        </p>
      </div>
      <div>
        <label
          className='text-xs mb-1 block'
          style={{ color: "var(--fg-secondary)" }}
        >
          Max length: {maxLength}
        </label>
        <input
          type='range'
          min={50}
          max={500}
          step={10}
          value={maxLength}
          onChange={(e) => {
            const v = Number(e.target.value);
            setMaxLength(v);
            if (minLength >= v) setMinLength(Math.max(20, v - 20));
          }}
          className='w-full slider-styled'
        />
      </div>
      <div>
        <label
          className='text-xs mb-1 block'
          style={{ color: "var(--fg-secondary)" }}
        >
          Min length: {minLength}
        </label>
        <input
          type='range'
          min={20}
          max={300}
          step={10}
          value={minLength}
          onChange={(e) => {
            const v = Number(e.target.value);
            setMinLength(v);
            if (maxLength <= v) setMaxLength(v + 20);
          }}
          className='w-full slider-styled'
        />
      </div>

      <button
        onClick={run}
        disabled={state.isFeatureLoading}
        className='w-full py-2 rounded-lg text-xs font-medium transition-colors'
        style={{
          background: "var(--accent)",
          color: "#fff",
          opacity: state.isFeatureLoading ? 0.5 : 1,
        }}
      >
        {state.isFeatureLoading
          ? "Generating audio…"
          : "Generate Audio Summary"}
      </button>

      {state.featureResult?.summary && (
        <>
          <div
            className='flex items-center gap-2 flex-wrap'
            style={{ fontSize: 10, color: "var(--fg-muted)" }}
          >
            <span
              className='px-1.5 py-0.5 rounded'
              style={{ background: "var(--bg-elevated)" }}
            >
              {state.featureResult.summarization_type}
            </span>
            <span>{state.featureResult.num_chunks_processed} chunks</span>
          </div>
          <div
            className='rounded-lg p-3 text-sm leading-relaxed'
            style={{
              background: "var(--bg-elevated)",
              border: "1px solid var(--border)",
              color: "var(--fg-primary)",
            }}
          >
            {state.featureResult.summary}
          </div>
        </>
      )}

      {state.featureResult?.audio_url && (
        <div
          className='flex items-center gap-3 rounded-lg p-3'
          style={{
            background: "var(--bg-elevated)",
            border: "1px solid var(--border)",
          }}
        >
          <button
            onClick={togglePlay}
            className='w-8 h-8 rounded-full flex items-center justify-center shrink-0'
            style={{ background: "var(--accent)" }}
          >
            {playing ? (
              <Pause size={14} color='#fff' />
            ) : (
              <Play size={14} color='#fff' className='ml-0.5' />
            )}
          </button>
          <audio
            ref={(el) => setAudioEl(el)}
            src={state.featureResult.audio_url}
            onEnded={() => setPlaying(false)}
            className='hidden'
          />
          <span
            className='text-xs truncate'
            style={{ color: "var(--fg-secondary)" }}
          >
            {state.featureResult.audio_filename || "audio.wav"}
          </span>
        </div>
      )}
    </div>
  );
}
