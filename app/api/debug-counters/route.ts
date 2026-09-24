import { NextResponse } from "next/server";
import { getCallCounters } from "../call-counters";

// Dev-only debug panel data source - see app/paint/page.tsx's DebugPanel.
// Cheap to leave reachable in production too (it's just in-memory counts,
// no user data), but the UI only renders it outside production.
export async function GET() {
  return NextResponse.json(getCallCounters());
}
