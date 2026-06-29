"use client";
import { Loader2 } from "lucide-react";

interface Props {
  url: string | null;
  loading: boolean;
  roomName: string;
}

export function SceneImage({ url, loading, roomName }: Props) {
  return (
    <div className="relative w-full flex-shrink-0" style={{ height: "38vh" }}>
      {/* Atmospheric placeholder */}
      <div className="absolute inset-0 image-placeholder" />

      {/* Scene image fades in over placeholder */}
      {url && (
        <img
          src={url}
          alt={roomName}
          className="absolute inset-0 w-full h-full object-cover transition-opacity duration-700"
        />
      )}

      {/* Generating overlay — clearly indicates a new image is being produced */}
      {loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-stone-950/55 backdrop-blur-[1px]">
          <Loader2 size={24} className="animate-spin text-amber-400" />
          <span className="text-xs text-stone-300 tracking-wide">Generating scene…</span>
        </div>
      )}

      {/* Room name caption */}
      {roomName && (
        <div className="absolute bottom-0 left-0 right-0 px-3 py-2 bg-gradient-to-t from-stone-950/80 to-transparent">
          <span className="text-xs text-stone-400 font-medium uppercase tracking-widest">{roomName}</span>
        </div>
      )}
    </div>
  );
}
