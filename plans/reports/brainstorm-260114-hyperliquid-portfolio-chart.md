# Brainstorm: Stacked Horizontal Bar Chart for Hyperliquid Portfolio

**Date:** 2026-01-14
**Status:** Agreed
**Topic:** Visualizing Hyperliquid portfolio data with stacked horizontal bar chart

---

## Problem Statement

Build a **Stacked Horizontal Bar Chart** to visualize Hyperliquid portfolio data, specifically showing the breakdown between **Perp** and **Spot** contributions for multiple metrics (Account Value, PnL, Volume) using the **"day"** and **"perpDay"** data from the portfolio API.

---

## Requirements

1. **Data Source:** Hyperliquid Info API - `POST https://api.hyperliquid.xyz/info`
   - Request: `{ "type": "portfolio", "user": "0x..." }`
   - Uses `day` (total) and `perpDay` (perp only) periods

2. **Metrics to Display:**
   - Account Value (Perp vs Spot breakdown)
   - PnL (Perp vs Spot breakdown)
   - Volume (Perp vs Spot breakdown)

3. **Time Period:** Day only (single snapshot)

4. **Calculation Logic:**
   - `perpValue` = value from `perpDay` object
   - `spotValue` = `day` value - `perpDay` value

---

## Evaluated Approaches

### Option A: Recharts (Recommended) ✅

```tsx
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';

<ResponsiveContainer width="100%" height={200}>
  <BarChart layout="vertical" data={chartData} margin={{ left: 100 }}>
    <XAxis type="number" />
    <YAxis type="category" dataKey="label" />
    <Tooltip />
    <Legend />
    <Bar dataKey="perpValue" stackId="a" fill="#3bb5d3" name="Perp" />
    <Bar dataKey="spotValue" stackId="a" fill="#7dd3fc" name="Spot" />
  </BarChart>
</ResponsiveContainer>
```

| Pros | Cons |
|------|------|
| Already installed (v2.15.4) | Larger bundle than pure CSS |
| Existing ChartContainer wrapper | Limited complex interactions |
| Built-in tooltips, legends | - |
| Responsive by default | - |
| Matches design system | - |

### Option B: Native CSS + Tailwind

| Pros | Cons |
|------|------|
| Zero dependencies | No built-in tooltips |
| Fastest rendering | Manual calculations |
| Full styling control | More code to maintain |

### Option C: Chart.js

| Pros | Cons |
|------|------|
| More flexibility | New dependency needed |
| Better animations | Different API pattern |

---

## Final Recommended Solution

**Use Recharts with `layout="vertical"` for horizontal stacked bars.**

### Data Structure

```typescript
interface PortfolioMetric {
  label: string;      // "Account Value" | "PnL" | "Volume"
  perpValue: number;  // Value from perpDay
  spotValue: number;  // Calculated: day - perpDay
  total: number;      // Sum for reference
}

// Example transformation from API response
function transformPortfolioData(dayData: PortfolioData, perpDayData: PortfolioData): PortfolioMetric[] {
  const dayValue = parseFloat(dayData.accountValueHistory.at(-1)?.[1] || "0");
  const perpValue = parseFloat(perpDayData.accountValueHistory.at(-1)?.[1] || "0");

  const dayPnl = parseFloat(dayData.pnlHistory.at(-1)?.[1] || "0");
  const perpPnl = parseFloat(perpDayData.pnlHistory.at(-1)?.[1] || "0");

  const dayVlm = parseFloat(dayData.vlm);
  const perpVlm = parseFloat(perpDayData.vlm);

  return [
    {
      label: "Account Value",
      perpValue: perpValue,
      spotValue: Math.max(0, dayValue - perpValue),
      total: dayValue,
    },
    {
      label: "PnL",
      perpValue: perpPnl,
      spotValue: dayPnl - perpPnl,
      total: dayPnl,
    },
    {
      label: "Volume",
      perpValue: perpVlm,
      spotValue: Math.max(0, dayVlm - perpVlm),
      total: dayVlm,
    },
  ];
}
```

### Visual Design

```
┌─────────────────────────────────────────────────────────────┐
│ Account Value  ████████████████████████████│███████│        │
│                     Perp (#3bb5d3)         │ Spot  │        │
│                                            │(#7dd3fc)       │
├─────────────────────────────────────────────────────────────┤
│ PnL            ████████████████████│████│                   │
│                     Perp           │Spot│                   │
├─────────────────────────────────────────────────────────────┤
│ Volume         █████████████████████████████████│██│        │
│                          Perp               │Spot│          │
└─────────────────────────────────────────────────────────────┘
```

### Color Scheme (from existing design system)

| Segment | Color | CSS Variable |
|---------|-------|--------------|
| Perp | `#3bb5d3` | `--chart-1` / `--primary` |
| Spot | `#7dd3fc` | `--chart-2` |

---

## Implementation Considerations

### API Integration

```typescript
// Hook for fetching portfolio data
async function fetchPortfolio(userAddress: string) {
  const response = await fetch("https://api.hyperliquid.xyz/info", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type: "portfolio", user: userAddress }),
  });
  return response.json();
}

// Extract day and perpDay from response
const portfolio = await fetchPortfolio("0x...");
const dayData = portfolio.find(([key]) => key === "day")?.[1];
const perpDayData = portfolio.find(([key]) => key === "perpDay")?.[1];
```

### Edge Cases

1. **Empty portfolio:** Show empty state with "No data" message
2. **Negative PnL:** Handle negative values (display correctly, may need different color)
3. **Zero spot activity:** If `day === perpDay`, spot segment = 0 (valid, just show full perp bar)
4. **API errors:** Graceful error handling with retry

### Enhancements (Future)

- Toggle between time periods (day/week/month/allTime)
- Add percentage labels on hover
- Animate on data load
- Export/share functionality

---

## Risks

| Risk | Mitigation |
|------|------------|
| API rate limiting | Cache responses, use React Query |
| Large numbers formatting | Use number formatter (e.g., `$1.2M`) |
| Negative values display | Handle with conditional styling |
| Mobile responsiveness | Use ResponsiveContainer, test on small screens |

---

## Success Metrics

1. Chart renders correctly with real API data
2. Perp vs Spot breakdown is visually clear
3. Tooltips show exact values
4. Responsive on all screen sizes
5. Loads within 500ms after API response

---

## Next Steps

1. [ ] Create `useHyperliquidPortfolio` hook with React Query
2. [ ] Build `PortfolioStackedBar` component
3. [ ] Add number formatting utility
4. [ ] Integrate into dashboard page
5. [ ] Add loading/error states
6. [ ] Test with real wallet addresses

---

## Dependencies

- **Existing:** `recharts@2.15.4`, `@tanstack/react-query@5.83.0`
- **New:** None required

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/hooks/useHyperliquidPortfolio.ts` | Create |
| `src/components/PortfolioStackedBar.tsx` | Create |
| `src/lib/formatters.ts` | Create (number formatting) |
| `src/pages/Dashboard.tsx` | Modify (add chart) |
