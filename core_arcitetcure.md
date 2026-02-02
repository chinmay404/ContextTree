Arc description, step by step, with no ambiguity.

Node A (root or any node):

Maintains rolling_summary_A (covers A’s past, compact)

Maintains raw_messages_A (last K messages only)

Old raw messages are folded into rolling_summary_A and discarded

This summary belongs only to A

Fork happens at message 3 (A → B):

At the moment of fork, you do a snapshot:

Take rolling_summary_A up to message 3

Take a small buffer of exact messages around the fork (e.g. message 3, maybe 1–2 before)

Create a new summary object for B from that snapshot

Node B starts with:

rolling_summary_B = snapshot(summary of A up to fork)

raw_messages_B = [buffer messages]

A is now irrelevant to B at runtime

Conversation continues in B:

New messages only affect rolling_summary_B

When B grows, you re-summarize:
“Previous summary + new messages → updated rolling_summary_B”

No write-back to A

No dependency on A

Fork again (B → C):

Same logic:

Snapshot rolling_summary_B (+ small delta if needed)

rolling_summary_C starts as “summary of A + B”

raw_messages_C starts near fork

C evolves independently

Key invariant across the entire tree:

Summaries accumulate forward

Ownership never flows backward or sideways

No node reads live state from another node

This guarantees:

No context leaks

No depth-based token growth

Deterministic behavior

That’s the full arc. What you described in your last message matches this exactly.