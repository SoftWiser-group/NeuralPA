[P1_Replace_Type]^private static final  int  serialVersionUID = 2824226005990582538L;^38^^^^^33^43^private static final long serialVersionUID = 2824226005990582538L;^[CLASS] Product   [VARIABLES] 
[P8_Replace_Mix]^private static final long serialVersionUID = 2824226005990582538;^38^^^^^33^43^private static final long serialVersionUID = 2824226005990582538L;^[CLASS] Product   [VARIABLES] 
[P1_Replace_Type]^private  short  n;^41^^^^^36^46^private long n;^[CLASS] Product   [VARIABLES] 
[P8_Replace_Mix]^private  int  n;^41^^^^^36^46^private long n;^[CLASS] Product   [VARIABLES] 
[P1_Replace_Type]^private int value;^46^^^^^41^51^private double value;^[CLASS] Product   [VARIABLES] 
[P3_Replace_Literal]^n = 9;^52^^^^^51^54^n = 0;^[CLASS] Product  [METHOD] <init> [RETURN_TYPE] Product()   [VARIABLES] double  value  long  n  serialVersionUID  boolean  
[P8_Replace_Mix]^n = 0 - 0;^52^^^^^51^54^n = 0;^[CLASS] Product  [METHOD] <init> [RETURN_TYPE] Product()   [VARIABLES] double  value  long  n  serialVersionUID  boolean  
[P8_Replace_Mix]^value ;^53^^^^^51^54^value = Double.NaN;^[CLASS] Product  [METHOD] <init> [RETURN_TYPE] Product()   [VARIABLES] double  value  long  n  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^value = d;value = Double.NaN;^53^^^^^51^54^value = Double.NaN;^[CLASS] Product  [METHOD] <init> [RETURN_TYPE] Product()   [VARIABLES] double  value  long  n  serialVersionUID  boolean  
[P2_Replace_Operator]^if  ( n < 0 )  {^60^^^^^59^66^if  ( n == 0 )  {^[CLASS] Product  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P3_Replace_Literal]^if  ( n == -2 )  {^60^^^^^59^66^if  ( n == 0 )  {^[CLASS] Product  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P5_Replace_Variable]^if  ( serialVersionUID == 0 )  {^60^^^^^59^66^if  ( n == 0 )  {^[CLASS] Product  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P8_Replace_Mix]^if  ( n == 2 )  {^60^^^^^59^66^if  ( n == 0 )  {^[CLASS] Product  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P15_Unwrap_Block]^value = d;^60^61^62^63^64^59^66^if  ( n == 0 )  { value = d; } else { value *= d; }^[CLASS] Product  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P16_Remove_Block]^^60^61^62^63^64^59^66^if  ( n == 0 )  { value = d; } else { value *= d; }^[CLASS] Product  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P5_Replace_Variable]^value *= value;^63^^^^^59^66^value *= d;^[CLASS] Product  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P8_Replace_Mix]^value *=  null;^63^^^^^59^66^value *= d;^[CLASS] Product  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^value = d;value *= d;^63^^^^^59^66^value *= d;^[CLASS] Product  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P5_Replace_Variable]^value = value;^61^^^^^59^66^value = d;^[CLASS] Product  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P8_Replace_Mix]^value =  null;^61^^^^^59^66^value = d;^[CLASS] Product  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^value *= d;value = d;^61^^^^^59^66^value = d;^[CLASS] Product  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^value = Double.NaN;value = d;^61^^^^^59^66^value = d;^[CLASS] Product  [METHOD] increment [RETURN_TYPE] void   final double d [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P5_Replace_Variable]^return d;^72^^^^^71^73^return value;^[CLASS] Product  [METHOD] getResult [RETURN_TYPE] double   [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P5_Replace_Variable]^return serialVersionUID;^79^^^^^78^80^return n;^[CLASS] Product  [METHOD] getN [RETURN_TYPE] long   [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P8_Replace_Mix]^value  =  value ;^86^^^^^85^88^value = Double.NaN;^[CLASS] Product  [METHOD] clear [RETURN_TYPE] void   [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^value = d;value = Double.NaN;^86^^^^^85^88^value = Double.NaN;^[CLASS] Product  [METHOD] clear [RETURN_TYPE] void   [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P8_Replace_Mix]^n = 3;^87^^^^^85^88^n = 0;^[CLASS] Product  [METHOD] clear [RETURN_TYPE] void   [VARIABLES] double  d  value  long  n  serialVersionUID  boolean  
[P1_Replace_Type]^int product = Double.NaN;^105^^^^^104^113^double product = Double.NaN;^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( values, length, length )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( values, begin, begin )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test (  begin, length )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( values,  length )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( values, begin )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( length, begin, values )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( values, length, begin )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P6_Replace_Expression]^if  ( begin + length )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P15_Unwrap_Block]^product = 1.0; for (int i = begin; i < (begin + length); i++) {    product *= values[i];};^106^107^108^109^110^104^113^if  ( test ( values, begin, length )  )  { product = 1.0; for  ( int i = begin; i < begin + length; i++ )  { product *= values[i]; }^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P16_Remove_Block]^^106^107^108^109^110^104^113^if  ( test ( values, begin, length )  )  { product = 1.0; for  ( int i = begin; i < begin + length; i++ )  { product *= values[i]; }^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P1_Replace_Type]^for  (  short  i = begin; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P2_Replace_Operator]^for  ( int i = begin; i <= begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P2_Replace_Operator]^for  ||  ( int i = begin; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^for  ( int i = length; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^for  ( beginnt i = i; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^for  ( lengthnt i = begin; i < begin + i; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P8_Replace_Mix]^product *=  null[i];^109^^^^^104^113^product *= values[i];^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P11_Insert_Donor_Statement]^product = 1.0;product *= values[i];^109^^^^^104^113^product *= values[i];^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P3_Replace_Literal]^product = 0.14285714285714285;^107^^^^^104^113^product = 1.0;^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P8_Replace_Mix]^product = 1.0D;^107^^^^^104^113^product = 1.0;^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P11_Insert_Donor_Statement]^product *= values[i];product = 1.0;^107^^^^^104^113^product = 1.0;^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P1_Replace_Type]^for  (  long  i = begin; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P2_Replace_Operator]^for  ==  ( int i = begin; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^for  ( lengthnt i = begin; i < begin + length; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^for  ( int i = begin; i < begin + begin; i++ )  {^108^^^^^104^113^for  ( int i = begin; i < begin + length; i++ )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P3_Replace_Literal]^product = 0.25;^107^^^^^104^113^product = 1.0;^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P8_Replace_Mix]^product = 3.0d;^107^^^^^104^113^product = 1.0;^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( values, i, length )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( values, begin, i )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^if  ( test ( begin, values, length )  )  {^106^^^^^104^113^if  ( test ( values, begin, length )  )  {^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
[P5_Replace_Variable]^return value;^112^^^^^104^113^return product;^[CLASS] Product  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length [VARIABLES] boolean  double[]  values  double  d  product  value  long  n  serialVersionUID  int  begin  i  length  
