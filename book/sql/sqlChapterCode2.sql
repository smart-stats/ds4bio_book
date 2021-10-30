.mode csv
.import county_pop_arcos.csv population
.import county_annual.csv annual
.import land_area.csv land
.tables

.mode column
pragma table_info(population);
.mode column
pragma table_info(annual);
.mode column
pragma table_info(land);
