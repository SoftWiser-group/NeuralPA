[BugLab_Wrong_Literal]^return new DateMidnight ( iYear, DateTimeConstants.JULY, iYear ) .toDateTime (  ) ;^47^48^^^^46^49^return new DateMidnight ( iYear, DateTimeConstants.JULY, 12 ) .toDateTime (  ) ;^[CLASS] GBAnniversaries 1 2 3 4 5  [METHOD] create [RETURN_TYPE] DateTime   int iYear [VARIABLES] AnniversaryFactory  JULY_12  MAY_DAY_BANK_HOLIDAY  SCOTTISH_SUMMER_BANK_HOLIDAY  SPRING_BANK_HOLIDAY  SUMMER_BANK_HOLIDAY  int  iYear  boolean  
[BugLab_Wrong_Literal]^return new DateMidnight ( iYear, DateTimeConstants.JULY, 13 ) .toDateTime (  ) ;^47^48^^^^46^49^return new DateMidnight ( iYear, DateTimeConstants.JULY, 12 ) .toDateTime (  ) ;^[CLASS] GBAnniversaries 1 2 3 4 5  [METHOD] create [RETURN_TYPE] DateTime   int iYear [VARIABLES] AnniversaryFactory  JULY_12  MAY_DAY_BANK_HOLIDAY  SCOTTISH_SUMMER_BANK_HOLIDAY  SPRING_BANK_HOLIDAY  SUMMER_BANK_HOLIDAY  int  iYear  boolean  
[BugLab_Wrong_Literal]^return new DateMidnight ( iYear, DateTimeConstants.JULY, iYear ) .toDateTime (  ) ;^47^48^^^^46^49^return new DateMidnight ( iYear, DateTimeConstants.JULY, 12 ) .toDateTime (  ) ;^[CLASS] 3  [METHOD] create [RETURN_TYPE] DateTime   int iYear [VARIABLES] boolean  int  iYear  
