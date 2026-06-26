"use client";
import { useState, useRef, useEffect } from "react";
import { Mic, MicOff, Send } from "lucide-react";

interface Props {
  onSubmit: (text: string) => void;
  disabled: boolean;
}

export function InputBar({ onSubmit, disabled }: Props) {
  const [text, setText] = useState("");
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-focus on desktop
  useEffect(() => {
    if (!disabled && inputRef.current) inputRef.current.focus();
  }, [disabled]);

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setText("");
  };

  const toggleRecording = async () => {
    if (recording) {
      mediaRecorderRef.current?.stop();
      setRecording(false);
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mr = new MediaRecorder(stream);
    const chunks: Blob[] = [];
    mr.ondataavailable = (e) => chunks.push(e.data);
    mr.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunks, { type: "audio/webm" });
      const form = new FormData();
      form.append("audio", blob, "audio.webm");
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/sessions/voice/transcribe`,
        { method: "POST", credentials: "include", body: form }
      );
      const { text: transcribed } = await res.json();
      if (transcribed) onSubmit(transcribed);
    };
    mr.start();
    mediaRecorderRef.current = mr;
    setRecording(true);
  };

  return (
    <div className="flex-shrink-0 px-4 pb-safe pb-4 pt-2 border-t border-stone-800 bg-stone-950">
      <div className="flex items-center gap-2">
        <input
          ref={inputRef}
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          disabled={disabled || recording}
          placeholder={disabled ? "..." : "What do you do?"}
          className="flex-1 bg-stone-900 border border-stone-700 rounded-lg px-4 py-2.5 text-stone-100 placeholder-stone-600 text-sm focus:outline-none focus:border-stone-500 disabled:opacity-50"
        />
        <button
          onClick={toggleRecording}
          disabled={disabled}
          className={`p-2.5 rounded-lg border transition-colors ${
            recording
              ? "bg-red-900 border-red-700 text-red-300"
              : "bg-stone-900 border-stone-700 text-stone-400 hover:text-stone-200"
          } disabled:opacity-40`}
        >
          {recording ? <MicOff size={18} /> : <Mic size={18} />}
        </button>
        <button
          onClick={submit}
          disabled={disabled || !text.trim()}
          className="p-2.5 rounded-lg bg-amber-700 hover:bg-amber-600 text-stone-100 transition-colors disabled:opacity-40"
        >
          <Send size={18} />
        </button>
      </div>
    </div>
  );
}
