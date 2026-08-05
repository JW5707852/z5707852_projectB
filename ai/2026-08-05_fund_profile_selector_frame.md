# Prompt log - Fund Profile selector frame

## What I wanted

Make the Fund Profile fund selector visibly recognisable as an interactive
control on the deployed app.

## Prompt(s)

"为fund选择的icon加上框框" (add a border around the fund-selection control).

## What the assistant produced

It placed the Fund Profile selector in a bordered panel and added a visible
border, rounded corners, background, and a divider before the dropdown-icon
area through the existing Streamlit CSS layer.

## What was wrong or risky

The selected fund appeared as unframed text in the deployed layout, so it was
not immediately clear that users could open it. CSS selectors can change when
Streamlit changes its DOM structure, so the bordered Streamlit container also
provides a resilient visible frame.

## What I changed and why

Student confirmation pending: review the deployed Fund Profile control and
confirm that the framed selector is clear without making the page feel heavy.
