# Annotator Profiles

Demographic and perceptual profile for each of the 187 crowd annotators.

## File

### `annotator_profiles.csv`

| Column | Description |
|--------|-------------|
| `annotator_id` | Anonymized identifier, e.g. `A001`–`A187` |
| `age` | Age group: `20s or younger`, `30s`, `40s`, `50s or older` |
| `gender` | `Male`, `Female` |
| `anime_familiarity` | Self-reported familiarity: `Low`, `Medium`, `High` |
| `category` | Perceptual category of free-form response (Gemini-classified) |
| `reason_en` | English translation of the classification rationale |
| `freeform_responses` | Free-form description(s) of "anime-like" voice (Japanese, as list) |

## Demographics

| Category | Group | Count |
|----------|-------|------:|
| Age | 20s or younger | 8 |
| | 30s | 48 |
| | 40s | 80 |
| | 50s or older | 51 |
| Gender | Male | 142 |
| | Female | 45 |
| Anime Familiarity | Low | 9 |
| | Medium | 103 |
| | High | 75 |

## Perceptual Categories

Free-form responses were classified by Gemini into five categories corresponding to the acoustic proxy features analyzed in the paper:

| Category | Count | Description |
|----------|------:|-------------|
| Emotional Explicitness | 62 | Exaggerated or vivid emotional expression |
| Timbre Difference | 48 | Distinctive voice quality or resonance |
| Prosodic Salience | 38 | Wide pitch range, dynamic intonation, or rhythm |
| Articulation Clarity | 34 | Clear and precise pronunciation |
| Temporal Control | 5 | Pacing, pausing, or speech rate |
