[P1_Replace_Type]^private static final  short  serialVersionUID = -8007759382851708045L;^37^^^^^32^42^private static final long serialVersionUID = -8007759382851708045L;^[CLASS] AbstractUnivariateStatistic   [VARIABLES] 
[P8_Replace_Mix]^private static final  int  serialVersionUID = -8007759382851708045;^37^^^^^32^42^private static final long serialVersionUID = -8007759382851708045L;^[CLASS] AbstractUnivariateStatistic   [VARIABLES] 
[P3_Replace_Literal]^test ( values, -3, -3 ) ;^43^^^^^42^45^test ( values, 0, 0 ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] evaluate [RETURN_TYPE] double   final double[] values [VARIABLES] double[]  values  long  serialVersionUID  boolean  
[P7_Replace_Invocation]^evaluate ( values, 0, 0 ) ;^43^^^^^42^45^test ( values, 0, 0 ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] evaluate [RETURN_TYPE] double   final double[] values [VARIABLES] double[]  values  long  serialVersionUID  boolean  
[P14_Delete_Statement]^^43^^^^^42^45^test ( values, 0, 0 ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] evaluate [RETURN_TYPE] double   final double[] values [VARIABLES] double[]  values  long  serialVersionUID  boolean  
[P3_Replace_Literal]^return evaluate ( values, 3, values.length ) ;^44^^^^^42^45^return evaluate ( values, 0, values.length ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] evaluate [RETURN_TYPE] double   final double[] values [VARIABLES] double[]  values  long  serialVersionUID  boolean  
[P5_Replace_Variable]^return evaluate (  0, values.length ) ;^44^^^^^42^45^return evaluate ( values, 0, values.length ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] evaluate [RETURN_TYPE] double   final double[] values [VARIABLES] double[]  values  long  serialVersionUID  boolean  
[P5_Replace_Variable]^return evaluate ( values, 0 ) ;^44^^^^^42^45^return evaluate ( values, 0, values.length ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] evaluate [RETURN_TYPE] double   final double[] values [VARIABLES] double[]  values  long  serialVersionUID  boolean  
[P5_Replace_Variable]^return evaluate ( values.length, 0, values ) ;^44^^^^^42^45^return evaluate ( values, 0, values.length ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] evaluate [RETURN_TYPE] double   final double[] values [VARIABLES] double[]  values  long  serialVersionUID  boolean  
[P7_Replace_Invocation]^return test ( values, 0, values.length ) ;^44^^^^^42^45^return evaluate ( values, 0, values.length ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] evaluate [RETURN_TYPE] double   final double[] values [VARIABLES] double[]  values  long  serialVersionUID  boolean  
[P3_Replace_Literal]^return evaluate ( values, 9, values.length ) ;^44^^^^^42^45^return evaluate ( values, 0, values.length ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] evaluate [RETURN_TYPE] double   final double[] values [VARIABLES] double[]  values  long  serialVersionUID  boolean  
[P14_Delete_Statement]^^44^^^^^42^45^return evaluate ( values, 0, values.length ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] evaluate [RETURN_TYPE] double   final double[] values [VARIABLES] double[]  values  long  serialVersionUID  boolean  
[P2_Replace_Operator]^if  ( values != null )  {^76^^^^^74^99^if  ( values == null )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P8_Replace_Mix]^if  ( values == true )  {^76^^^^^74^99^if  ( values == null )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("input value array is null");^76^77^78^^^74^99^if  ( values == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P16_Remove_Block]^^76^77^78^^^74^99^if  ( values == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P13_Insert_Block]^if  ( values == null )  {     throw new IllegalArgumentException ( "input value array is null" ) ; }^77^^^^^74^99^[Delete]^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P11_Insert_Donor_Statement]^throw new IllegalArgumentException ( "begin + length > values.length" ) ;throw new IllegalArgumentException  (" ")  ;^77^^^^^74^99^throw new IllegalArgumentException  (" ")  ;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P13_Insert_Block]^if  ( length < 0 )  {     throw new IllegalArgumentException ( "length cannot be negative" ) ; }^77^^^^^74^99^[Delete]^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P13_Insert_Block]^if  ( begin < 0 )  {     throw new IllegalArgumentException ( "start position cannot be negative" ) ; }^77^^^^^74^99^[Delete]^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P2_Replace_Operator]^if  ( begin > 0 )  {^80^^^^^74^99^if  ( begin < 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P3_Replace_Literal]^if  ( begin < -9 )  {^80^^^^^74^99^if  ( begin < 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P5_Replace_Variable]^if  ( length < 0 )  {^80^^^^^74^99^if  ( begin < 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P8_Replace_Mix]^if  ( begin < 0 * 0 )  {^80^^^^^74^99^if  ( begin < 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P9_Replace_Statement]^if  ( length == 0 )  {^80^^^^^74^99^if  ( begin < 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("start position cannot be negative");^80^81^82^^^74^99^if  ( begin < 0 )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P16_Remove_Block]^^80^81^82^^^74^99^if  ( begin < 0 )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P4_Replace_Constructor]^throw throw  new IllegalArgumentException ( "length cannot be negative" )   ;^81^^^^^74^99^throw new IllegalArgumentException  (" ")  ;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P13_Insert_Block]^if  ( length < 0 )  {     throw new IllegalArgumentException ( "length cannot be negative" ) ; }^81^^^^^74^99^[Delete]^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P13_Insert_Block]^if  ( begin < 0 )  {     throw new IllegalArgumentException ( "start position cannot be negative" ) ; }^81^^^^^74^99^[Delete]^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P8_Replace_Mix]^throw new IllegalArgumentException ( "begin + length > values.length" ) ; ;^81^^^^^74^99^throw new IllegalArgumentException  (" ")  ;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P11_Insert_Donor_Statement]^throw new IllegalArgumentException ( "begin + length > values.length" ) ;throw new IllegalArgumentException  (" ")  ;^81^^^^^74^99^throw new IllegalArgumentException  (" ")  ;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P8_Replace_Mix]^return ;^81^^^^^74^99^throw new IllegalArgumentException  (" ")  ;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P2_Replace_Operator]^if  ( length <= 0 )  {^84^^^^^74^99^if  ( length < 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P3_Replace_Literal]^if  ( length < -6 )  {^84^^^^^74^99^if  ( length < 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P5_Replace_Variable]^if  ( begin < 0 )  {^84^^^^^74^99^if  ( length < 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P9_Replace_Statement]^if  ( length == 0 )  {^84^^^^^74^99^if  ( length < 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("length cannot be negative");^84^85^86^^^74^99^if  ( length < 0 )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P16_Remove_Block]^^84^85^86^^^74^99^if  ( length < 0 )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P4_Replace_Constructor]^throw throw  new IllegalArgumentException ( "start position cannot be negative" )   ;^85^^^^^74^99^throw new IllegalArgumentException  (" ")  ;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P13_Insert_Block]^if  ( length < 0 )  {     throw new IllegalArgumentException ( "length cannot be negative" ) ; }^85^^^^^74^99^[Delete]^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P13_Insert_Block]^if  ( begin < 0 )  {     throw new IllegalArgumentException ( "start position cannot be negative" ) ; }^85^^^^^74^99^[Delete]^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P8_Replace_Mix]^throw new IllegalArgumentException ( "begin + length > values.length" ) ; ;^85^^^^^74^99^throw new IllegalArgumentException  (" ")  ;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P11_Insert_Donor_Statement]^throw new IllegalArgumentException ( "begin + length > values.length" ) ;throw new IllegalArgumentException  (" ")  ;^85^^^^^74^99^throw new IllegalArgumentException  (" ")  ;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P2_Replace_Operator]^if  ( begin + length >= values.length )  {^88^^^^^74^99^if  ( begin + length > values.length )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P2_Replace_Operator]^if  ( begin  <  length > values.length )  {^88^^^^^74^99^if  ( begin + length > values.length )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P5_Replace_Variable]^if  ( begin + begin > values.length )  {^88^^^^^74^99^if  ( begin + length > values.length )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P5_Replace_Variable]^if  ( begin + length > length )  {^88^^^^^74^99^if  ( begin + length > values.length )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P5_Replace_Variable]^if  ( length + begin > values.length )  {^88^^^^^74^99^if  ( begin + length > values.length )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P5_Replace_Variable]^if  ( begin + values > length.length )  {^88^^^^^74^99^if  ( begin + length > values.length )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P5_Replace_Variable]^if  ( begin + length > values )  {^88^^^^^74^99^if  ( begin + length > values.length )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P6_Replace_Expression]^if  ( begin + length )  {^88^^^^^74^99^if  ( begin + length > values.length )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P8_Replace_Mix]^if  ( begin + length > begin )  {^88^^^^^74^99^if  ( begin + length > values.length )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("begin + length > values.length");^88^89^90^91^^74^99^if  ( begin + length > values.length )  { throw new IllegalArgumentException ( "begin + length > values.length" ) ; }^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P16_Remove_Block]^^88^89^90^91^^74^99^if  ( begin + length > values.length )  { throw new IllegalArgumentException ( "begin + length > values.length" ) ; }^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P13_Insert_Block]^if  ( length < 0 )  {     throw new IllegalArgumentException ( "length cannot be negative" ) ; }^88^^^^^74^99^[Delete]^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P13_Insert_Block]^if  ( begin < 0 )  {     throw new IllegalArgumentException ( "start position cannot be negative" ) ; }^88^^^^^74^99^[Delete]^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P13_Insert_Block]^if  ( values == null )  {     throw new IllegalArgumentException ( "input value array is null" ) ; }^88^^^^^74^99^[Delete]^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P3_Replace_Literal]^throw new IllegalArgumentException ( "l" ) ;^89^90^^^^74^99^throw new IllegalArgumentException ( "begin + length > values.length" ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P13_Insert_Block]^if  (  ( begin + length )  >  ( values.length )  )  {     throw new IllegalArgumentException ( "begin + length > values.length" ) ; }^89^^^^^74^99^[Delete]^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P3_Replace_Literal]^throw new IllegalArgumentException ( "begin + length > values.lengthgin " ) ;^89^90^^^^74^99^throw new IllegalArgumentException ( "begin + length > values.length" ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P11_Insert_Donor_Statement]^throw new IllegalArgumentException  (" ")  ;throw new IllegalArgumentException ( "begin + length > values.length" ) ;^89^90^^^^74^99^throw new IllegalArgumentException ( "begin + length > values.length" ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P13_Insert_Block]^if  ( length < 0 )  {     throw new IllegalArgumentException ( "length cannot be negative" ) ; }^89^^^^^74^99^[Delete]^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P13_Insert_Block]^if  ( begin < 0 )  {     throw new IllegalArgumentException ( "start position cannot be negative" ) ; }^89^^^^^74^99^[Delete]^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P3_Replace_Literal]^throw new IllegalArgumentException ( " > valuegin + length > values.length" ) ;^89^90^^^^74^99^throw new IllegalArgumentException ( "begin + length > values.length" ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P3_Replace_Literal]^throw new IllegalArgumentException ( "begin + length > values.lengthin + length > values." ) ;^89^90^^^^74^99^throw new IllegalArgumentException ( "begin + length > values.length" ) ;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P2_Replace_Operator]^if  ( length != 0 )  {^93^^^^^74^99^if  ( length == 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P3_Replace_Literal]^if  ( length == 6 )  {^93^^^^^74^99^if  ( length == 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P5_Replace_Variable]^if  ( begin == 0 )  {^93^^^^^74^99^if  ( length == 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P8_Replace_Mix]^if  ( length == 0  )  {^93^^^^^74^99^if  ( length == 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P9_Replace_Statement]^if  ( length < 0 )  {^93^^^^^74^99^if  ( length == 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P9_Replace_Statement]^if  ( begin < 0 )  {^93^^^^^74^99^if  ( length == 0 )  {^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P15_Unwrap_Block]^return false;^93^94^95^^^74^99^if  ( length == 0 )  { return false; }^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P16_Remove_Block]^^93^94^95^^^74^99^if  ( length == 0 )  { return false; }^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P3_Replace_Literal]^return true;^94^^^^^74^99^return false;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
[P3_Replace_Literal]^return false;^97^^^^^74^99^return true;^[CLASS] AbstractUnivariateStatistic  [METHOD] test [RETURN_TYPE] boolean   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  long  serialVersionUID  int  begin  length  
