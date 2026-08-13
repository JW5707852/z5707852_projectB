# Prompt log - sentiment heatmap contrast

## What I wanted

Make the sector-sentiment heatmap easier to interpret by increasing the visual
contrast of positive and negative sentiment.

## Prompt(s)

The heatmap isn't very intuitive—could the color changes be made more pronounced?

## What the assistant produced

It changed the display-only heatmap scale to use the 95th percentile of the
absolute 21-day scores instead of the single largest score, added a steeper
red–white–green diverging palette, and labelled the legend as more negative,
neutral, and more positive. It retained the underlying scores in hover text.

## What was wrong or risky

Scaling to the absolute maximum allowed a small number of extreme observations
to make typical historical differences look almost white. A percentile-based
display scale deliberately saturates rare extremes, so the chart must state
that it is a visual scaling decision and must not be used as a new model input.

## What I changed and why

I requested stronger colour changes because the maximum-based scale made most
historical regimes look nearly neutral. I accepted the 95th-percentile
display-only scale and steeper red–white–green palette because the local preview
made ordinary variation clearer while hover text retained the unmodified score.
I also retained the limitation that rare extremes saturate visually; no
sentiment value, model output, or portfolio rule was changed.
