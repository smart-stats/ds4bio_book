.mode csv
.import county_pop_arcos.csv population
.import county_annual.csv annual
.import land_area.csv land
.tables
pragma table_info(population);
pragma table_info(annual);
pragma table_info(land);
select BUYER_COUNTY, BUYER_STATE, STATE, COUNTY, year, population from population limit 5;
select * from annual where countyfips = "NA" limit 10;
