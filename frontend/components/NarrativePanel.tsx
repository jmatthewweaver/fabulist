"use client";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface Props {
  text: string;
  isStreaming: boolean;
  isConnecting: boolean;
  turnIndex: number;
  totalTurns: number;
  onPrev: () => void;
  onNext: () => void;
}

export function NarrativePanel({ text, isStreaming, isConnecting, turnIndex, totalTurns, onPrev, onNext }: Props) {
  const canPrev = turnIndex > 0;
  const canNext = turnIndex < totalTurns - 1;

  return (
    <div className="flex-1 flex flex-col min-h-0 px-5 py-4">
      {/* Narrative text — scrollable if very long, but showing just the current turn */}
      <div className="flex-1 overflow-y-auto">
        {isConnecting ? (
          <div className="flex items-center gap-2 text-stone-500 text-sm mt-2">
            <span className="inline-flex gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-stone-500 animate-bounce [animation-delay:0ms]" />
              <span className="w-1.5 h-1.5 rounded-full bg-stone-500 animate-bounce [animation-delay:150ms]" />
              <span className="w-1.5 h-1.5 rounded-full bg-stone-500 animate-bounce [animation-delay:300ms]" />
            </span>
            <span>Entering the world...</span>
          </div>
        ) : (
          <p className="narrative-text text-stone-200 whitespace-pre-wrap">
            {text}
            {isStreaming && <span className="animate-pulse text-stone-500">▍</span>}
          </p>
        )}
      </div>

      {/* Page controls */}
      {totalTurns > 1 && (
        <div className="flex items-center justify-between mt-3 pt-3 border-t border-stone-800">
          <button
            onClick={onPrev}
            disabled={!canPrev}
            className="flex items-center gap-1 text-xs text-stone-500 hover:text-stone-300 disabled:opacity-30 transition-colors"
          >
            <ChevronLeft size={14} /> prev
          </button>
          <span className="text-xs text-stone-600">
            {turnIndex + 1} / {totalTurns}
          </span>
          <button
            onClick={onNext}
            disabled={!canNext}
            className="flex items-center gap-1 text-xs text-stone-500 hover:text-stone-300 disabled:opacity-30 transition-colors"
          >
            next <ChevronRight size={14} />
          </button>
        </div>
      )}
    </div>
  );
}
