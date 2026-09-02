"use client";

import { FormEvent, useState } from "react";

type ModelPrediction = {
  label: string;
  confidence: number;
};

type CompareResponse = {
  rnn: ModelPrediction;
  distilbert: ModelPrediction;
};
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/predict/compare`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text,
        }),
      });

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const data: CompareResponse = await response.json();

      setResult(data);
    } catch {
      setError("Could not connect to the prediction API.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#212121] text-[#ececec]">
      <div className="mx-auto max-w-6xl px-6 py-16 md:py-20">
        <section className="max-w-4xl">
          <div className="inline-flex rounded-full border border-[#3f3f3f] bg-[#2a2a2a] px-3 py-1 text-xs font-medium text-[#b4b4b4]">
            NLP · Model Comparison
          </div>

          <h1 className="mt-6 text-4xl font-semibold tracking-tight md:text-6xl">
            Support Ticket Intent Classifier
          </h1>

          <p className="mt-5 max-w-3xl text-base leading-7 text-[#b4b4b4] md:text-lg">
            Compare a vanilla RNN trained from scratch with a fine-tuned
            DistilBERT Transformer on the same customer support message.
          </p>
        </section>

        <form onSubmit={handleSubmit} className="mt-10">
          <div className="rounded-3xl border border-[#3f3f3f] bg-[#2f2f2f] p-3 shadow-sm">
            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Example: Where is my package?"
              rows={5}
              className="w-full resize-none bg-transparent px-3 py-3 text-lg text-[#ececec] outline-none placeholder:text-[#8e8e8e]"
            />

            <div className="flex justify-end border-t border-[#3f3f3f] pt-3">
              <button
                type="submit"
                disabled={loading || !text.trim()}
                className="rounded-xl bg-[#10a37f] px-6 py-3 font-semibold text-white transition hover:bg-[#0d8f70] disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loading ? "Comparing..." : "Compare Models"}
              </button>
            </div>
          </div>
        </form>

        {result && (
          <section className="mt-12">
            <div className="mb-6 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <h2 className="text-2xl font-semibold">Prediction Results</h2>

              <p className="text-sm text-[#8e8e8e]">
                Same input · Two architectures
              </p>
            </div>

            <div className="grid gap-5 md:grid-cols-2">
              <ModelCard
                title="Vanilla RNN"
                subtitle="Trained from scratch"
                prediction={result.rnn}
                challengeAccuracy="56.79%"
              />

              <ModelCard
                title="DistilBERT"
                subtitle="Pretrained Transformer · Fine-tuned"
                prediction={result.distilbert}
                challengeAccuracy="90.12%"
                featured
              />
            </div>

            {result.rnn.label !== result.distilbert.label && (
              <div className="mt-5 rounded-2xl border border-[#3f3f3f] bg-[#2a2a2a] px-5 py-4 text-sm leading-6 text-[#b4b4b4]">
                The models disagree on this example. The RNN learns only from
                the task dataset, while DistilBERT starts with pretrained
                language representations and is then fine-tuned for the 27
                support intents.
              </div>
            )}
          </section>
        )}

        {error && (
          <div className="mt-8 rounded-2xl border border-red-900/70 bg-red-950/30 px-5 py-4 text-red-200">
            {error}
          </div>
        )}

        <section className="mt-16 border-t border-[#3a3a3a] pt-8">
          <div className="grid gap-8 text-sm sm:grid-cols-3">
            <InfoBlock
              title="Vanilla RNN"
              text="Custom recurrent neural network trained from scratch."
            />

            <InfoBlock
              title="DistilBERT"
              text="Pretrained Transformer fine-tuned for 27 support intents."
            />

            <InfoBlock
              title="Serving"
              text="FastAPI backend serving both models through one comparison endpoint."
            />
          </div>
        </section>
      </div>
    </main>
  );
}

function ModelCard({
  title,
  subtitle,
  prediction,
  challengeAccuracy,
  featured = false,
}: {
  title: string;
  subtitle: string;
  prediction: ModelPrediction;
  challengeAccuracy: string;
  featured?: boolean;
}) {
  const confidence = prediction.confidence * 100;

  return (
    <div
      className={`rounded-2xl border p-6 ${
        featured
          ? "border-[#10a37f]/50 bg-[#2f2f2f]"
          : "border-[#3f3f3f] bg-[#2f2f2f]"
      }`}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-xl font-semibold">{title}</h3>

            {featured && (
              <span className="rounded-full bg-[#10a37f]/15 px-2 py-1 text-[11px] font-semibold text-[#19c99a]">
                Transformer
              </span>
            )}
          </div>

          <p className="mt-1 text-sm text-[#8e8e8e]">{subtitle}</p>
        </div>

        <span className="whitespace-nowrap rounded-full border border-[#4a4a4a] bg-[#212121] px-3 py-1 text-xs font-medium text-[#b4b4b4]">
          Challenge {challengeAccuracy}
        </span>
      </div>

      <div className="mt-9">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#8e8e8e]">
          Predicted intent
        </p>

        <p className="mt-3 break-words text-3xl font-semibold tracking-tight">
          {prediction.label}
        </p>
      </div>

      <div className="mt-9">
        <div className="flex items-center justify-between">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#8e8e8e]">
            Confidence
          </p>

          <p className="text-xl font-semibold">{confidence.toFixed(2)}%</p>
        </div>

        <div className="mt-3 h-2 overflow-hidden rounded-full bg-[#444444]">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              featured ? "bg-[#10a37f]" : "bg-[#ececec]"
            }`}
            style={{
              width: `${confidence}%`,
            }}
          />
        </div>
      </div>
    </div>
  );
}

function InfoBlock({ title, text }: { title: string; text: string }) {
  return (
    <div>
      <p className="font-semibold text-[#ececec]">{title}</p>

      <p className="mt-2 leading-6 text-[#8e8e8e]">{text}</p>
    </div>
  );
}
