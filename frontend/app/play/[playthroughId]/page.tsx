// Screen 3: Game Session — the main play interface
"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import { SceneImage } from "../../../components/SceneImage";
import { NarrativePanel } from "../../../components/NarrativePanel";
import { InputBar } from "../../../components/InputBar";

interface Turn {
  id: number;
  userInput: string;
  narrative: string;
  room: string;
}

export default function PlayPage() {
  const { playthroughId } = useParams<{ playthroughId: string }>();

  const wsRef = useRef<WebSocket | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [currentTurn, setCurrentTurn] = useState(0);
  const [streamingText, setStreamingText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnecting, setIsConnecting] = useState(true);
  const [currentRoom, setCurrentRoom] = useState("");
  const [turn, setTurn] = useState(0);
  const [score, setScore] = useState<number | null>(null);
  const [maxScore, setMaxScore] = useState<number | null>(null);
  const [sceneDescription, setSceneDescription] = useState("");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageLoading, setImageLoading] = useState(false);
  const pendingNarrativeRef = useRef<string[]>([]);

  useEffect(() => {
    connectWs();
    return () => wsRef.current?.close();
  }, [playthroughId]);

  // Follow the latest turn as new ones arrive (manual paging still works between turns).
  useEffect(() => {
    setCurrentTurn(Math.max(0, turns.length - 1));
  }, [turns.length]);

  const connectWs = () => {
    const wsUrl = process.env.NEXT_PUBLIC_API_URL!.replace(/^http/, "ws");
    const ws = new WebSocket(`${wsUrl}/api/playthroughs/${playthroughId}/play`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      switch (msg.type) {
        case "narrative_chunk":
          setIsConnecting(false);
          pendingNarrativeRef.current.push(msg.text);
          setStreamingText((t) => t + msg.text);
          setIsStreaming(true);
          break;
        case "narrative_done": {
          const full = pendingNarrativeRef.current.join("");
          pendingNarrativeRef.current = [];
          setStreamingText("");
          setIsStreaming(false);
          setTurns((prev) => {
            // Fill the pending command turn (the last one still awaiting its narrative);
            // otherwise append — e.g. the opening scene, which has no preceding command.
            if (prev.length && prev[prev.length - 1].narrative === "") {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              updated[updated.length - 1] = { ...last, narrative: full, room: currentRoom };
              return updated;
            }
            return [...prev, { id: prev.length, userInput: "", narrative: full, room: currentRoom }];
          });
          break;
        }
        case "game_state":
          setCurrentRoom(msg.room);
          if (typeof msg.turn === "number") setTurn(msg.turn);
          setScore(typeof msg.score === "number" ? msg.score : null);
          setMaxScore(typeof msg.max_score === "number" ? msg.max_score : null);
          break;
        case "scene_description":
          setSceneDescription(msg.description || "");
          setImageLoading(true);   // the image for this scene is on its way
          break;
        case "image_ready":
          setImageUrl(`${process.env.NEXT_PUBLIC_API_URL}${msg.url}`);
          setImageLoading(false);
          break;
        case "error":
          setTurns((prev) => [
            ...prev,
            { id: prev.length, userInput: "", narrative: `[${msg.message}]`, room: currentRoom },
          ]);
          setIsStreaming(false);
          setIsConnecting(false);
          break;
      }
    };
  };

  const sendCommand = useCallback((text: string) => {
    if (!wsRef.current || isStreaming) return;
    wsRef.current.send(JSON.stringify({ type: "command", text }));
    setTurns((prev) => [
      ...prev,
      { id: prev.length, userInput: text, narrative: "", room: currentRoom },
    ]);
  }, [isStreaming, currentRoom]);

  const requestImage = useCallback(() => {
    if (!wsRef.current) return;
    setImageLoading(true);
    wsRef.current.send(JSON.stringify({ type: "request_image" }));
  }, []);

  const displayedTurn = turns[currentTurn];
  const narrativeToShow = isStreaming
    ? streamingText
    : displayedTurn?.narrative ?? "";

  return (
    <div className="flex flex-col h-[100dvh] max-w-lg mx-auto">
      <div className="flex items-center justify-between gap-3 px-3 py-2 text-xs border-b border-stone-800 bg-stone-950 shrink-0">
        <span className="truncate text-stone-300">{currentRoom || "…"}</span>
        <span className="flex gap-3 tabular-nums text-stone-400 shrink-0">
          <span>Turn {turn}</span>
          {score !== null && (
            <span>Score {score}{maxScore ? ` / ${maxScore}` : ""}</span>
          )}
        </span>
      </div>
      <SceneImage url={imageUrl} loading={imageLoading} roomName={currentRoom} onRequestImage={requestImage} />
      <NarrativePanel
        sceneDescription={sceneDescription}
        command={displayedTurn?.userInput ?? ""}
        text={narrativeToShow}
        isStreaming={isStreaming}
        isConnecting={isConnecting}
        turnIndex={currentTurn}
        totalTurns={turns.length}
        onPrev={() => setCurrentTurn((i) => Math.max(0, i - 1))}
        onNext={() => setCurrentTurn((i) => Math.min(turns.length - 1, i + 1))}
      />
      <InputBar onSubmit={sendCommand} disabled={isStreaming || isConnecting} />
    </div>
  );
}
