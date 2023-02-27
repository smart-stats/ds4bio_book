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
update annual set countyfips = 05097 where BUYER_STATE = "AR" and BUYER_COUNTY = "MONTGOMERY";
select * from annual where BUYER_STATE = "AR" and BUYER_COUNTY = "MONTGOMERY";
select * from annual where BUYER_COUNTY = "NA";
delete from annual where BUYER_COUNTY = "NA";
select * from annual where BUYER_COUNTY = "NA";
create table land_area as select Areaname, STCOU, LND110210D from land;
alter table land_area rename column STCOU to countyfips;
create table county_info as select * from population left join land_area using(countyfips);

