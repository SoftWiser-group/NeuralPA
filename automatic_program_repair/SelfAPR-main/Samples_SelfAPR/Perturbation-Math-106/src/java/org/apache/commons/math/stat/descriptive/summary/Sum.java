[P1_Replace_Type]^private static final  int  serialVersionUID = -8231831954703408316L;^38^^^^^33^43^private static final long serialVersionUID = -8231831954703408316L;^[CLASS] Sum   [VARIABLES] 
[P8_Replace_Mix]^private static final long serialVersionUID = -8231831954703408316;^38^^^^^33^43^private static final long serialVersionUID = -8231831954703408316L;^[CLASS] Sum   [VARIABLES] 
[P1_Replace_Type]^private  short  n;^41^^^^^36^46^private long n;^[CLASS] Sum   [VARIABLES] 
[P1_Replace_Type]^private int value;^46^^^^^41^51^private double value;^[CLASS] Sum   [VARIABLES] 
[P8_Replace_Mix]^n = 0 - 0;^52^^^^^51^54^n = 0;^[CLASS] Sum  [METHOD] <init> [RETURN_TYPE] Sum()   [VARIABLES] double  value  long  n  serialVersionUID  boolean  
[P8_Replace_Mix]^value  =  value ;^53^^^^^51^54^value = Double.NaN;^[CLASS] Sum  [METHOD] <init> [RETURN_TYPE] Sum()   [VARIABLES] double  value  long  n  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^value = d;value = Double.NaN;^53^^^^^51^54^value = Double.NaN;^[CLASS] Sum  [METHOD] <init> [RETURN_TYPE] Sum()   [VARIABLES] double  value  long  n  serialVersionUID  boolean  
[P2_Replace_Operator]^if  ( n != 0 )  {^60^^^^^59^66^if  ( n == 0 )  {^[CLASS] Sum  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P3_Replace_Literal]^if  ( n == -5 )  {^60^^^^^59^66^if  ( n == 0 )  {^[CLASS] Sum  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P5_Replace_Variable]^if  ( serialVersionUID == 0 )  {^60^^^^^59^66^if  ( n == 0 )  {^[CLASS] Sum  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P8_Replace_Mix]^if  ( n == 2 )  {^60^^^^^59^66^if  ( n == 0 )  {^[CLASS] Sum  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P15_Unwrap_Block]^value = d;^60^61^62^63^64^59^66^if  ( n == 0 )  { value = d; } else { value += d; }^[CLASS] Sum  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P16_Remove_Block]^^60^61^62^63^64^59^66^if  ( n == 0 )  { value = d; } else { value += d; }^[CLASS] Sum  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P5_Replace_Variable]^value += value;^63^^^^^59^66^value += d;^[CLASS] Sum  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P8_Replace_Mix]^value +=  null;^63^^^^^59^66^value += d;^[CLASS] Sum  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^value = d;value += d;^63^^^^^59^66^value += d;^[CLASS] Sum  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P5_Replace_Variable]^value = value;^61^^^^^59^66^value = d;^[CLASS] Sum  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P8_Replace_Mix]^value =  null;^61^^^^^59^66^value = d;^[CLASS] Sum  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^value += d;value = d;^61^^^^^59^66^value = d;^[CLASS] Sum  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^value = Double.NaN;value = d;^61^^^^^59^66^value = d;^[CLASS] Sum  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P5_Replace_Variable]^return d;^72^^^^^71^73^return value;^[CLASS] Sum  [METHOD] getResult [RETURN_TYPE] double   [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P5_Replace_Variable]^return serialVersionUID;^79^^^^^78^80^return n;^[CLASS] Sum  [METHOD] getN [RETURN_TYPE] long   [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P8_Replace_Mix]^value  =  value ;^86^^^^^85^88^value = Double.NaN;^[CLASS] Sum  [METHOD] clear [RETURN_TYPE] void   [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^value = d;value = Double.NaN;^86^^^^^85^88^value = Double.NaN;^[CLASS] Sum  [METHOD] clear [RETURN_TYPE] void   [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P8_Replace_Mix]^n = 1;^87^^^^^85^88^n = 0;^[CLASS] Sum  [METHOD] clear [RETURN_TYPE] void   [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P1_Replace_Type]^int sum = Double.NaN;^105^^^^^104^113^double sum = Double.NaN;^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( values, length, length )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( values, begin, begin )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test (  begin, length )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( values,  length )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( values, begin )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( begin, values, length )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P6_Replace_Expression]^if  ( begin + length )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P15_Unwrap_Block]^sum = 0.0; for (int i = begin; i < (begin + length); i++) {    sum += values[i];};^106^107^108^109^110^104^113^if  ( test ( values, begin, length )  )  { sum = 0.0; for  ( int i = begin; i < begin + length; i++ )  { sum += values[i]; }^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P16_Remove_Block]^^106^107^108^109^110^104^113^if  ( test ( values, begin, length )  )  { sum = 0.0; for  ( int i = begin; i < begin + length; i++ )  { sum += values[i]; }^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P1_Replace_Type]^for  (  short  i = begin; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P2_Replace_Operator]^for  ( int i = begin; i <= begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P2_Replace_Operator]^for   instanceof   ( int i = begin; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P2_Replace_Operator]^for  >  ( int i = begin; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P3_Replace_Literal]^for  ( int i = 0; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^for  ( int i = length; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^for  ( lengthnt i = begin; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^for  ( int i = begin; i < begin + begin; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^for  ( int i = begin; i < values.length ; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^for  ( beginnt i = i; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P8_Replace_Mix]^for  ( int i = 0; i < length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P8_Replace_Mix]^for  ( int i = 0; i < values.length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P8_Replace_Mix]^sum +=  null[i];^109^^^^^104^113^sum += values[i];^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P8_Replace_Mix]^sum = 1.0d;^107^^^^^104^113^sum = 0.0;^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P1_Replace_Type]^for  (  long  i = begin; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P2_Replace_Operator]^for  ( int i = begin; i > begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P2_Replace_Operator]^for  &  ( int i = begin; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P2_Replace_Operator]^for  <<  ( int i = begin; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^for  ( lengthnt i = begin; i < begin + i; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P8_Replace_Mix]^for  ( lengthnt i = begin; i < length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P8_Replace_Mix]^for  ( lengthnt i = 0; i < length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P8_Replace_Mix]^sum = 2.0d;^107^^^^^104^113^sum = 0.0;^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( values, i, length )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( values, begin, i )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( length, begin, values )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( values, length, begin )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^return value;^112^^^^^104^113^return sum;^[CLASS] Sum  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  sum  value  long  n  serialVersionUID  int  begin  i  length  