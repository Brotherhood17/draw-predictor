const TIER_ORDER = ["1", "2", "3"];
const TIER_COLOR_CLASS = { "1": "tier-1", "2": "tier-2", "3": "tier-3" };
const CONFIDENCE_CLASS = { "High": "conf-high", "Medium": "conf-medium", "Low": "conf-low" };

async function loadPredictions() {
  const container = document.getElementById("tiers");
  const updatedEl = document.getElementById("updated-at");

  try {
    const res = await fetch("predictions.json", { cache: "no-store" });
    const data = await res.json();

    if (data.updated_at) {
      const d = new Date(data.updated_at);
      let line = `Last updated ${d.toLocaleString()}`;
      if (typeof data.league_draw_rate === "number") {
        line += ` &middot; league average draw rate: ${(data.league_draw_rate * 100).toFixed(0)}%`;
      }
      updatedEl.innerHTML = line;
    }

    container.innerHTML = "";

    TIER_ORDER.forEach((tierKey) => {
      const tier = data.tiers && data.tiers[tierKey];
      const section = document.createElement("section");
      section.className = `tier ${TIER_COLOR_CLASS[tierKey]}`;
      section.style.animationDelay = `${(parseInt(tierKey) - 1) * 0.1}s`;

      if (!tier) {
        section.innerHTML = `
          <div class="tier-header">
            <h2>Tier ${tierKey}</h2>
          </div>
          <div class="locked">Not released yet &mdash; check back later today.</div>
        `;
        container.appendChild(section);
        return;
      }

      const cards = tier.picks.map((p) => {
        const confClass = CONFIDENCE_CLASS[p.confidence] || "conf-medium";
        const factors = (p.analysis || []).map((f) => `<li>${f}</li>`).join("");
        const baselineNote = typeof p.vs_baseline === "number"
          ? `<span class="vs-baseline">${p.vs_baseline >= 0 ? "+" : ""}${(p.vs_baseline * 100).toFixed(0)}pp vs league average</span>`
          : "";

        return `
          <div class="card">
            <div class="card-top">
              <div>
                <div class="matchup">${p.home} vs ${p.away}</div>
                <div class="match-date">${p.date}</div>
              </div>
              <div class="prob-block">
                <div class="prob">${(p.draw_probability * 100).toFixed(0)}%</div>
                <span class="confidence-badge ${confClass}">${p.confidence || ""} confidence</span>
              </div>
            </div>
            ${factors ? `<ul class="analysis">${factors}</ul>` : ""}
            ${baselineNote}
          </div>
        `;
      }).join("");

      section.innerHTML = `
        <div class="tier-header">
          <h2>${tier.label}</h2>
          <span class="tier-time">released ${new Date(tier.released_at).toLocaleTimeString()}</span>
        </div>
        ${cards}
      `;
      container.appendChild(section);
    });
  } catch (err) {
    container.innerHTML = `<p style="color:#8b909c;text-align:center;">Couldn't load predictions yet. ${err}</p>`;
  }
}

loadPredictions();
