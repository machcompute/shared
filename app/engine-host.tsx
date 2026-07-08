"use client";

import { useEffect } from "react";
import { attachProtocol } from "@/lib/protocol";

export default function EngineHost() {
  useEffect(() => attachProtocol(window), []);
  return null;
}
