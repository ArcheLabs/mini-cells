from __future__ import annotations

import argparse

from .clm_release import CLM


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text with MiniCells CLM-0.1.")
    parser.add_argument("model", help="Path to a CLM-0.1 release bundle")
    parser.add_argument("prompt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--routing", action="store_true")
    args = parser.parse_args()
    model = CLM.from_pretrained(args.model, device=args.device)
    result = model.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        return_routing=args.routing,
    )
    if args.routing:
        print(result.text)
        print("\nRouting usage by generation step:")
        for index, usage in enumerate(result.routing_usage):
            print(f"{index:03d}: {usage}")
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
