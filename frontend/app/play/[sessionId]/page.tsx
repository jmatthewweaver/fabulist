// Screen 3: Game Session — the main play interface
"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { useParams, useSearchParams } from "next/navigation";
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
  const { sessionId } = useParams<{ sessionId: string }>();
  const searchParams = useSearchParams();
  const restoreSaveId = searchParams.get("restore");

  const wsRef = useRef<WebSocket | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [currentTurn, setCurrentTurn] = useState(0);
  const [streamingText, setStreamingText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentRoom, setCurrentRoom] = useState("");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageLoading, setImageLoading] = useState(false);
  const pendingNarrativeRef = useRef<string[]>([]);

  useEffect(() => {
    // Restore save if needed, then connect WebSocket
    const init = async () => {
      if (restoreSaveId) {
        await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/api/sessions/${sessionId}/restore/${restoreSaveId}`,
          { method: "POST", credentials: "include" }
        );
      }
      connectWs();
    };
    init();
    return () => wsRef.current?.close();
  }, [sessionId]);

  const connectWs = () => {
    const wsUrl = process.env.NEXT_PUBLIC_API_URL!.replace(/^http/, "ws");
    const ws = new WebSocket(`${wsUrl}/api/sessions/${sessionId}/play`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      switch (msg.type) {
        case "narrative_chunk":
          pendingNarrativeRef.current.push(msg.text);
          setStreamingText((t) => t + msg.text);
          setIsStreaming(true);
          break;
        case "narrative_done": {
          const full = pendingNarrativeRef.current.join("");
          pendingNarrativeRef.current = [];
          setStreamingText("");
          setIsStreaming(false);
          setTurns((prev) => [
            ...prev,
            { id: prev.length, userInput: "", narrative: full, room: currentRoom },
          ]);
          break;
        }
        case "game_state":
          setCurrentRoom(msg.room);
          break;
        case "image_ready":
          setImageUrl(msg.url);
          setImageLoading(false);
          break;
        case "error":
          setTurns((prev) => [
            ...prev,
            { id: prev.length, userInput: "", narrative: `[${msg.message}]`, room: currentRoom },
          ]);
          setIsStreaming(false);
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
      {/* Hero image */}
      <SceneImage url={imageUrl} loading={imageLoading} roomName={currentRoom} onRequestImage={requestImage} />

      {/* Narrative */}
      <NarrativePanel
        text={narrativeToShow}
        isStreaming={isStreaming}
        turnIndex={currentTurn}
        totalTurns={turns.length}
        onPrev={() => setCurrentTurn((i) => Math.max(0, i - 1))}
        onNext={() => setCurrentTurn((i) => Math.min(turns.length - 1, i + 1))}
      />

      {/* Input */}
      <InputBar onSubmit={sendCommand} disabled={isStreaming} />
    </div>
  );
}
