[BugLab_Variable_Misuse]^this.mean = variance;^63^^^^^60^69^this.mean = mean;^[CLASS] StatisticalSummaryValues  [METHOD] <init> [RETURN_TYPE] StatisticalSummaryValues(double,double,long,double,double,double)   double mean double variance long n double max double min double sum [VARIABLES] double  max  mean  min  sum  variance  long  n  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^this.variance = sum;^64^^^^^60^69^this.variance = variance;^[CLASS] StatisticalSummaryValues  [METHOD] <init> [RETURN_TYPE] StatisticalSummaryValues(double,double,long,double,double,double)   double mean double variance long n double max double min double sum [VARIABLES] double  max  mean  min  sum  variance  long  n  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^this.n = serialVersionUID;^65^^^^^60^69^this.n = n;^[CLASS] StatisticalSummaryValues  [METHOD] <init> [RETURN_TYPE] StatisticalSummaryValues(double,double,long,double,double,double)   double mean double variance long n double max double min double sum [VARIABLES] double  max  mean  min  sum  variance  long  n  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^this.max = variance;^66^^^^^60^69^this.max = max;^[CLASS] StatisticalSummaryValues  [METHOD] <init> [RETURN_TYPE] StatisticalSummaryValues(double,double,long,double,double,double)   double mean double variance long n double max double min double sum [VARIABLES] double  max  mean  min  sum  variance  long  n  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^this.min = variance;^67^^^^^60^69^this.min = min;^[CLASS] StatisticalSummaryValues  [METHOD] <init> [RETURN_TYPE] StatisticalSummaryValues(double,double,long,double,double,double)   double mean double variance long n double max double min double sum [VARIABLES] double  max  mean  min  sum  variance  long  n  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^this.sum = variance;^68^^^^^60^69^this.sum = sum;^[CLASS] StatisticalSummaryValues  [METHOD] <init> [RETURN_TYPE] StatisticalSummaryValues(double,double,long,double,double,double)   double mean double variance long n double max double min double sum [VARIABLES] double  max  mean  min  sum  variance  long  n  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return variance;^75^^^^^74^76^return max;^[CLASS] StatisticalSummaryValues  [METHOD] getMax [RETURN_TYPE] double   [VARIABLES] double  max  mean  min  sum  variance  long  n  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return variance;^82^^^^^81^83^return mean;^[CLASS] StatisticalSummaryValues  [METHOD] getMean [RETURN_TYPE] double   [VARIABLES] double  max  mean  min  sum  variance  long  n  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return serialVersionUID;^96^^^^^95^97^return n;^[CLASS] StatisticalSummaryValues  [METHOD] getN [RETURN_TYPE] long   [VARIABLES] double  max  mean  min  sum  variance  long  n  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return Math.sqrt ( sum ) ;^110^^^^^109^111^return Math.sqrt ( variance ) ;^[CLASS] StatisticalSummaryValues  [METHOD] getStandardDeviation [RETURN_TYPE] double   [VARIABLES] double  max  mean  min  sum  variance  long  n  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return Math.sqrt ( min ) ;^110^^^^^109^111^return Math.sqrt ( variance ) ;^[CLASS] StatisticalSummaryValues  [METHOD] getStandardDeviation [RETURN_TYPE] double   [VARIABLES] double  max  mean  min  sum  variance  long  n  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return sum;^117^^^^^116^118^return variance;^[CLASS] StatisticalSummaryValues  [METHOD] getVariance [RETURN_TYPE] double   [VARIABLES] double  max  mean  min  sum  variance  long  n  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( object >= this  )  {^129^^^^^128^142^if  ( object == this  )  {^[CLASS] StatisticalSummaryValues  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] StatisticalSummaryValues  stat  Object  object  boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^130^^^^^128^142^return true;^[CLASS] StatisticalSummaryValues  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] StatisticalSummaryValues  stat  Object  object  boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( object instanceof StatisticalSummaryValues < false )  {^132^^^^^128^142^if  ( object instanceof StatisticalSummaryValues == false )  {^[CLASS] StatisticalSummaryValues  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] StatisticalSummaryValues  stat  Object  object  boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( object  <  StatisticalSummaryValues == false )  {^132^^^^^128^142^if  ( object instanceof StatisticalSummaryValues == false )  {^[CLASS] StatisticalSummaryValues  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] StatisticalSummaryValues  stat  Object  object  boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  
[BugLab_Wrong_Literal]^if  ( object instanceof StatisticalSummaryValues == true )  {^132^^^^^128^142^if  ( object instanceof StatisticalSummaryValues == false )  {^[CLASS] StatisticalSummaryValues  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] StatisticalSummaryValues  stat  Object  object  boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^133^^^^^128^142^return false;^[CLASS] StatisticalSummaryValues  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] StatisticalSummaryValues  stat  Object  object  boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  
[BugLab_Wrong_Operator]^return  ( MathUtils.equals ( stat.getMax (  ) , this.getMax (  )  )  || MathUtils.equals ( stat.getMean (  ) ,this.getMean (  )  )  && MathUtils.equals ( stat.getMin (  ) ,this.getMin (  )  )  && MathUtils.equals ( stat.getN (  ) , this.getN (  )  )  &&^136^137^138^139^^128^142^return  ( MathUtils.equals ( stat.getMax (  ) , this.getMax (  )  )  && MathUtils.equals ( stat.getMean (  ) ,this.getMean (  )  )  && MathUtils.equals ( stat.getMin (  ) ,this.getMin (  )  )  && MathUtils.equals ( stat.getN (  ) , this.getN (  )  )  &&^[CLASS] StatisticalSummaryValues  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] StatisticalSummaryValues  stat  Object  object  boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  
[BugLab_Wrong_Operator]^int result = 31 + MathUtils.hash ( getMax (  >=  )  ) ;^150^^^^^149^157^int result = 31 + MathUtils.hash ( getMax (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Literal]^int result = result + MathUtils.hash ( getMax (  )  ) ;^150^^^^^149^157^int result = 31 + MathUtils.hash ( getMax (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Operator]^result = result * 31 + MathUtils.hash ( getMean (  >>  )  ) ;^151^^^^^149^157^result = result * 31 + MathUtils.hash ( getMean (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Operator]^result = result + 31 + MathUtils.hash ( getMean (  )  ) ;^151^^^^^149^157^result = result * 31 + MathUtils.hash ( getMean (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Literal]^result = result *  + MathUtils.hash ( getMean (  )  ) ;^151^^^^^149^157^result = result * 31 + MathUtils.hash ( getMean (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Operator]^result = result * 31 + MathUtils.hash ( getMin (  ==  )  ) ;^152^^^^^149^157^result = result * 31 + MathUtils.hash ( getMin (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Operator]^result = result - 31 + MathUtils.hash ( getMin (  )  ) ;^152^^^^^149^157^result = result * 31 + MathUtils.hash ( getMin (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Literal]^result = result * 30 + MathUtils.hash ( getMin (  )  ) ;^152^^^^^149^157^result = result * 31 + MathUtils.hash ( getMin (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Operator]^result = result * 31 + MathUtils.hash ( getN (  &  )  ) ;^153^^^^^149^157^result = result * 31 + MathUtils.hash ( getN (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Operator]^result = result - 31 + MathUtils.hash ( getN (  )  ) ;^153^^^^^149^157^result = result * 31 + MathUtils.hash ( getN (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Literal]^result = result * result + MathUtils.hash ( getN (  )  ) ;^153^^^^^149^157^result = result * 31 + MathUtils.hash ( getN (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Operator]^result = result * 31 + MathUtils.hash ( getSum (  ^  )  ) ;^154^^^^^149^157^result = result * 31 + MathUtils.hash ( getSum (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Operator]^result = result / 31 + MathUtils.hash ( getSum (  )  ) ;^154^^^^^149^157^result = result * 31 + MathUtils.hash ( getSum (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Literal]^result = result * 30 + MathUtils.hash ( getSum (  )  ) ;^154^^^^^149^157^result = result * 31 + MathUtils.hash ( getSum (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Operator]^result = result * 31 + MathUtils.hash ( getVariance (  &&  )  ) ;^155^^^^^149^157^result = result * 31 + MathUtils.hash ( getVariance (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  
[BugLab_Wrong_Operator]^result = result / 31 + MathUtils.hash ( getVariance (  )  ) ;^155^^^^^149^157^result = result * 31 + MathUtils.hash ( getVariance (  )  ) ;^[CLASS] StatisticalSummaryValues  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  max  mean  min  sum  variance  long  n  serialVersionUID  int  result  