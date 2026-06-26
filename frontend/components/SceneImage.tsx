"use client";
import { Camera } from "lucide-react";

interface Props {
  url: string | null;
  loading: boolean;
  roomName: string;
  onRequestImage: () => void;
}

export function SceneImage({ url, loading, roomName, onRequestImage }: Props) {
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

      {/* 📷 button — always available, bypasses cooldown */}
      <button
        onClick={onRequestImage}
        disabled={loading}
        className="absolute top-2 right-2 p-1.5 rounded-md bg-stone-950/60 hover:bg-stone-950/80 text-stone-400 hover:text-stone-200 transition-colors disabled:opacity-40"
        title="Generate image of current scene"
      >
        <Camera size={15} />
      </button>

      {/* Loading indicator */}
      {loading && (
        <div className="absolute top-2 left-2 w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
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
