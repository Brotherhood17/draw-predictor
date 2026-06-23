// Optional serverless endpoint: GET /api/predictions
//
// The simplest version of this site just fetches /predictions.json as a
// static file (that's what public/app.js does by default) - you don't
// need this function at all to get started.
//
// This is here as a starting point for later, if you want to add
// server-side logic - e.g. filtering by league, reading from a database
// like Supabase instead of a flat file, hiding tiers until a specific
// time server-side, etc.

const fs = require("fs");
const path = require("path");

exports.handler = async function (event) {
  try {
    const filePath = path.join(__dirname, "..", "..", "public", "predictions.json");
    const raw = fs.readFileSync(filePath, "utf-8");
    const data = JSON.parse(raw);

    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    };
  } catch (err) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: err.message }),
    };
  }
};
