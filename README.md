# Paintx.ai

AI-powered paint visualization platform. Upload a photo of a house and preview different paint colours on it before buying paint.

## Overview

Paintx.ai lets a user upload a photo of a building (interior or exterior) and see how it would look repainted in different colours, using automated wall/surface detection and AI-based rendering rather than a flat digital overlay.

## Tech stack

Next.js, TypeScript, Tailwind CSS, OpenCV.js, Replicate API, Sharp. Deployed on Vercel.

## Engineering highlights

Request deduplication: fixed a React Strict Mode race condition that was causing duplicate AI API calls (2 per upload instead of 1) by building an idempotent, ID-based request-deduplication system, cutting redundant calls by 50%.

Wall detection coverage: reached 99% building-surface coverage by architecting a fallback-recovery pipeline that reclaims paintable area two segmentation models (SAM2, SAM3) individually missed.

Cost optimization: benchmarked Replicate model pricing and output quality directly, then migrated the rendering pipeline to a lower-cost model with no quality loss.

Colour rendering: built a custom pixel-level, luminance-preserving colour-blend algorithm to avoid flat, artificial-looking paint output, verified against real photo data pixel-by-pixel.

## Getting started

Clone the repo, run npm install, set up the required environment variables locally, then run npm run dev.

## Environment variables

Requires a Replicate API token and related service keys, kept in a local .env.local file which is gitignored and never committed.

## Live demo

https://colourvista.vercel.app

## Project status

Live and in active development.

## Author

Arshpreet Singh Bhasin - GitHub: https://github.com/arshpreetbhasin1-collab
