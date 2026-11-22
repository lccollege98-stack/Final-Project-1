# Diamond Dataset Dictionary

| Column | Meaning | Type |
|---|---|---|
| carat | Diamond weight in carats | Numeric |
| cut | Cut quality: Fair to Ideal | Categorical |
| color | Color grade: D (best) to J | Categorical |
| clarity | Internal/external flaw grade: I1 to IF | Categorical |
| depth | Total depth percentage | Numeric |
| table | Table width percentage | Numeric |
| x | Length in millimeters | Numeric |
| y | Width in millimeters | Numeric |
| z | Depth in millimeters | Numeric |
| price | Listed price in US dollars | Numeric |

## Engineered Features

| Feature | Formula or Meaning |
|---|---|
| volume | x * y * z |
| surface_area_proxy | x * y |
| cut_score | Ordinal cut quality score |
| color_score | Ordinal color quality score |
| clarity_score | Ordinal clarity quality score |
| premium_score | Sum of cut, color, and clarity scores |
| carat_volume_ratio | Carat divided by volume |
