# Data Dictionary

The dataset has the following columns:

- **carat**: weight of the diamond
- **cut**: cut quality, from Fair to Ideal
- **color**: colour grade, from D to J
- **clarity**: clarity grade, from I1 to IF
- **depth**: total depth percentage
- **table**: width of the top of the diamond
- **x**: length in millimetres
- **y**: width in millimetres
- **z**: depth in millimetres
- **price**: selling price in US dollars

## New columns created during the project

- **volume**: x multiplied by y and z
- **surface_area_proxy**: x multiplied by y
- **cut_score**: number given to each cut category
- **color_score**: number given to each color category
- **clarity_score**: number given to each clarity category
- **premium_score**: cut score, color score and clarity score added together
- **carat_volume_ratio**: carat divided by volume
