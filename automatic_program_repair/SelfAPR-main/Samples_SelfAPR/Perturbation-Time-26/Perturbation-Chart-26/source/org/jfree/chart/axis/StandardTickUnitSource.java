[P1_Replace_Type]^private static final int LOG_10_VALUE = Math.log ( 10.0 ) ;^54^^^^^49^59^private static final double LOG_10_VALUE = Math.log ( 10.0 ) ;^[CLASS] StandardTickUnitSource   [VARIABLES] 
[P3_Replace_Literal]^private static final double LOG_10_VALUE = Math.log ( 70.0 ) ;^54^^^^^49^59^private static final double LOG_10_VALUE = Math.log ( 10.0 ) ;^[CLASS] StandardTickUnitSource   [VARIABLES] 
[P7_Replace_Invocation]^private static final double LOG_10_VALUE = Math.ceil ( 10.0 ) ;^54^^^^^49^59^private static final double LOG_10_VALUE = Math.log ( 10.0 ) ;^[CLASS] StandardTickUnitSource   [VARIABLES] 
[P8_Replace_Mix]^private static final double LOG_10_VALUE ;^54^^^^^49^59^private static final double LOG_10_VALUE = Math.log ( 10.0 ) ;^[CLASS] StandardTickUnitSource   [VARIABLES] 
[P1_Replace_Type]^int x = unit.getSize (  ) ;^64^^^^^63^69^double x = unit.getSize (  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P14_Delete_Statement]^^64^^^^^63^69^double x = unit.getSize (  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P1_Replace_Type]^int log = Math.log ( x )  / LOG_10_VALUE;^65^^^^^63^69^double log = Math.log ( x )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P2_Replace_Operator]^double log = Math.log ( x )  - LOG_10_VALUE;^65^^^^^63^69^double log = Math.log ( x )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P5_Replace_Variable]^double log = Math.log ( higher )  / LOG_10_VALUE;^65^^^^^63^69^double log = Math.log ( x )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P5_Replace_Variable]^double log = Math.log ( x )  / x;^65^^^^^63^69^double log = Math.log ( x )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P5_Replace_Variable]^double log = Math.log ( LOG_10_VALUE )  / x;^65^^^^^63^69^double log = Math.log ( x )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P7_Replace_Invocation]^double log = Math.ceil ( x )  / LOG_10_VALUE;^65^^^^^63^69^double log = Math.log ( x )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P7_Replace_Invocation]^double log = Math .pow ( x , LOG_10_VALUE )   / LOG_10_VALUE;^65^^^^^63^69^double log = Math.log ( x )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P11_Insert_Donor_Statement]^double log = Math.log ( size )  / LOG_10_VALUE;double log = Math.log ( x )  / LOG_10_VALUE;^65^^^^^63^69^double log = Math.log ( x )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P8_Replace_Mix]^double log = Math.log ( log )  / LOG_10_VALUE;^65^^^^^63^69^double log = Math.log ( x )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P14_Delete_Statement]^^65^^^^^63^69^double log = Math.log ( x )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P1_Replace_Type]^int higher = Math.ceil ( log ) ;^66^^^^^63^69^double higher = Math.ceil ( log ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P5_Replace_Variable]^double higher = Math.ceil ( x ) ;^66^^^^^63^69^double higher = Math.ceil ( log ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P7_Replace_Invocation]^double higher = Math.log ( log ) ;^66^^^^^63^69^double higher = Math.ceil ( log ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P14_Delete_Statement]^^66^^^^^63^69^double higher = Math.ceil ( log ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P3_Replace_Literal]^return new NumberTickUnit ( Math.pow ( 3, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^67^68^^^^63^69^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P5_Replace_Variable]^return new NumberTickUnit ( Math.pow ( 10, x ) , new DecimalFormat ( "0.0E0" )  ) ;^67^68^^^^63^69^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P7_Replace_Invocation]^return new NumberTickUnit ( Math .log ( LOG_10_VALUE )  , new DecimalFormat ( "0.0E0" )  ) ;^67^68^^^^63^69^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P3_Replace_Literal]^return new NumberTickUnit ( Math.pow ( 1, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^67^68^^^^63^69^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P3_Replace_Literal]^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0..0E0" )  ) ;^67^68^^^^63^69^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P3_Replace_Literal]^new DecimalFormat ( ".0E0" )  ) ;^68^^^^^63^69^new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P3_Replace_Literal]^return new NumberTickUnit ( Math.pow ( 19, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^67^68^^^^63^69^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P7_Replace_Invocation]^return new NumberTickUnit ( Math .log ( higher )  , new DecimalFormat ( "0.0E0" )  ) ;^67^68^^^^63^69^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P14_Delete_Statement]^^67^68^^^^63^69^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P7_Replace_Invocation]^return getCeilingTickUnit ( unit ) ;^80^^^^^79^81^return getLargerTickUnit ( unit ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P14_Delete_Statement]^^80^^^^^79^81^return getLargerTickUnit ( unit ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] TickUnit  unit  double  LOG_10_VALUE  higher  log  x  boolean  
[P1_Replace_Type]^int log = Math.log ( size )  / LOG_10_VALUE;^92^^^^^91^96^double log = Math.log ( size )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P2_Replace_Operator]^double log = Math.log ( size )  * LOG_10_VALUE;^92^^^^^91^96^double log = Math.log ( size )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P5_Replace_Variable]^double log = Math.log ( size )  / x;^92^^^^^91^96^double log = Math.log ( size )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P5_Replace_Variable]^double log = Math.log ( LOG_10_VALUE )  / size;^92^^^^^91^96^double log = Math.log ( size )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P7_Replace_Invocation]^double log = Math.ceil ( size )  / LOG_10_VALUE;^92^^^^^91^96^double log = Math.log ( size )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P8_Replace_Mix]^double log = Math.log ( x )  / LOG_10_VALUE;^92^^^^^91^96^double log = Math.log ( size )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P11_Insert_Donor_Statement]^double log = Math.log ( x )  / LOG_10_VALUE;double log = Math.log ( size )  / LOG_10_VALUE;^92^^^^^91^96^double log = Math.log ( size )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P8_Replace_Mix]^double log = Math.log ( log )  / LOG_10_VALUE;^92^^^^^91^96^double log = Math.log ( size )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P14_Delete_Statement]^^92^^^^^91^96^double log = Math.log ( size )  / LOG_10_VALUE;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P1_Replace_Type]^int higher = Math.ceil ( log ) ;^93^^^^^91^96^double higher = Math.ceil ( log ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P5_Replace_Variable]^double higher = Math.ceil ( x ) ;^93^^^^^91^96^double higher = Math.ceil ( log ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P7_Replace_Invocation]^double higher = Math.log ( log ) ;^93^^^^^91^96^double higher = Math.ceil ( log ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P5_Replace_Variable]^double higher = Math.ceil ( size ) ;^93^^^^^91^96^double higher = Math.ceil ( log ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P14_Delete_Statement]^^93^^^^^91^96^double higher = Math.ceil ( log ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P3_Replace_Literal]^return new NumberTickUnit ( Math.pow ( 17, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^94^95^^^^91^96^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P3_Replace_Literal]^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0." )  ) ;^94^95^^^^91^96^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P5_Replace_Variable]^return new NumberTickUnit ( Math.pow ( 10, x ) , new DecimalFormat ( "0.0E0" )  ) ;^94^95^^^^91^96^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P8_Replace_Mix]^return new NumberTickUnit ( Math .log ( x )  , new DecimalFormat ( "0.0E0" )  ) ;^94^95^^^^91^96^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P3_Replace_Literal]^return new NumberTickUnit ( Math.pow ( 6, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^94^95^^^^91^96^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P3_Replace_Literal]^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0" )  ) ;^94^95^^^^91^96^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P5_Replace_Variable]^return new NumberTickUnit ( Math.pow ( 10, size ) , new DecimalFormat ( "0.0E0" )  ) ;^94^95^^^^91^96^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P3_Replace_Literal]^new DecimalFormat ( "0.0E" )  ) ;^95^^^^^91^96^new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P3_Replace_Literal]^return new NumberTickUnit ( Math.pow ( 3, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^94^95^^^^91^96^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P7_Replace_Invocation]^return new NumberTickUnit ( Math .log ( log )  , new DecimalFormat ( "0.0E0" )  ) ;^94^95^^^^91^96^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  
[P14_Delete_Statement]^^94^95^^^^91^96^return new NumberTickUnit ( Math.pow ( 10, higher ) , new DecimalFormat ( "0.0E0" )  ) ;^[CLASS] StandardTickUnitSource  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   double size [VARIABLES] double  LOG_10_VALUE  higher  log  size  x  boolean  