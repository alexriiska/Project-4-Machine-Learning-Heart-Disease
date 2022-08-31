ERD setup and queries


CREATE TABLE "heart_failure" (
    "Age" int NOT NULL,
    "Sex" char(1)   NOT NULL,
    "ChestPainType" varchar  NOT NULL,
    "RestingBP" int   NOT NULL,
    "Cholesterol" int   NOT NULL,
    "FastingBS" int   NOT NULL,
    "RestingECG" varchar   NOT NULL,
	"MaxHR" int NOT NULL,
	"ExerciseAngina" varchar NOT NULL,
	"OldPeak" float NOT NULL,
	"ST_Slope" varchar NOT NULL,
	"HeartDisease" int NOT NULL
);

#Loading in data
COPY heart_failure
FROM 'C:\Users\Danny Dao\Desktop\UCI PROGRAM\HOMEWORK\Project-4\Resources\heart.csv'
DELIMITER ','
CSV HEADER;