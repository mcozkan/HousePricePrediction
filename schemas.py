import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import Column, Integer, String, Float
import sqlalchemy

###############################################################################################################################################################
# Pydantic Models: HousePrice
###############################################################################################################################################################

class HousePrice(BaseModel):
    Id: int = Field(...)
    MSSubClass: int
    MSZoning: Optional[str] = None
    LotFrontage: float
    LotArea: int
    Street: str
    Alley: str | float | None= None
    LotShape: str
    LandContour: str
    Utilities: Optional[str] = None
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: Optional[str] = None
    Condition2: Optional[str] = None
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int = Field(...)
    YearRemodAdd: int
    RoofStyle: str = Field(...)
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: str
    MasVnrArea: float
    ExterQual: str
    ExterCond: str
    Foundation: Optional[str] = None
    BsmtQual: str
    BsmtCond: str
    BsmtExposure: str
    BsmtFinType1: str
    BsmtFinSF1: int
    BsmtFinType2: str
    BsmtFinSF2: int
    BsmtUnfSF: int
    TotalBsmtSF: int
    Heating: str = Field(...)
    HeatingQC: str
    CentralAir: str
    Electrical: str
    fisrt_1stFlrSF: int
    second_2ndFlrSF: int
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: int
    BsmtHalfBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: str | float | None= None
    GarageType: str
    GarageYrBlt: float
    GarageFinish: str
    GarageCars: int
    GarageArea: int
    GarageQual: str
    GarageCond: str
    PavedDrive: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    third_3SsnPorch: int
    ScreenPorch: int
    PoolArea: Optional[int] = None
    PoolQC: str | float | None= None
    Fence: str | float | None= None
    MiscFeature: str | float | None= None
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: str = Field(...)
    SaleCondition: str
    SalePrice: int = Field(...)

###############################################################################################################################################################
# SQLAlchemy Models: HousePrice
###############################################################################################################################################################

