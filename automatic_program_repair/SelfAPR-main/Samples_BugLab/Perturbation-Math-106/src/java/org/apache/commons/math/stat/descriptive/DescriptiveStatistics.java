[BugLab_Wrong_Literal]^public static final int INFINITE_WINDOW = -0;^81^^^^^76^86^public static final int INFINITE_WINDOW = -1;^[CLASS] DescriptiveStatistics   [VARIABLES] 
[BugLab_Wrong_Operator]^if  ( getN (  )  >= 0 )  {^124^^^^^122^132^if  ( getN (  )  > 0 )  {^[CLASS] DescriptiveStatistics  [METHOD] getStandardDeviation [RETURN_TYPE] double   [VARIABLES] boolean  long  serialVersionUID  int  INFINITE_WINDOW  double  stdDev  
[BugLab_Wrong_Operator]^if  ( getN (  )  < 0 )  {^124^^^^^122^132^if  ( getN (  )  > 0 )  {^[CLASS] DescriptiveStatistics  [METHOD] getStandardDeviation [RETURN_TYPE] double   [VARIABLES] boolean  long  serialVersionUID  int  INFINITE_WINDOW  double  stdDev  
[BugLab_Wrong_Literal]^if  ( getN (  )  > INFINITE_WINDOW )  {^124^^^^^122^132^if  ( getN (  )  > 0 )  {^[CLASS] DescriptiveStatistics  [METHOD] getStandardDeviation [RETURN_TYPE] double   [VARIABLES] boolean  long  serialVersionUID  int  INFINITE_WINDOW  double  stdDev  
[BugLab_Wrong_Operator]^if  ( getN (  )  < 1 )  {^125^^^^^122^132^if  ( getN (  )  > 1 )  {^[CLASS] DescriptiveStatistics  [METHOD] getStandardDeviation [RETURN_TYPE] double   [VARIABLES] boolean  long  serialVersionUID  int  INFINITE_WINDOW  double  stdDev  
[BugLab_Wrong_Literal]^if  ( getN (  )  > INFINITE_WINDOW )  {^125^^^^^122^132^if  ( getN (  )  > 1 )  {^[CLASS] DescriptiveStatistics  [METHOD] getStandardDeviation [RETURN_TYPE] double   [VARIABLES] boolean  long  serialVersionUID  int  INFINITE_WINDOW  double  stdDev  
[BugLab_Wrong_Operator]^if  ( getN (  )  >= 1 )  {^125^^^^^122^132^if  ( getN (  )  > 1 )  {^[CLASS] DescriptiveStatistics  [METHOD] getStandardDeviation [RETURN_TYPE] double   [VARIABLES] boolean  long  serialVersionUID  int  INFINITE_WINDOW  double  stdDev  
[BugLab_Wrong_Literal]^if  ( getN (  )  > 0 )  {^125^^^^^122^132^if  ( getN (  )  > 1 )  {^[CLASS] DescriptiveStatistics  [METHOD] getStandardDeviation [RETURN_TYPE] double   [VARIABLES] boolean  long  serialVersionUID  int  INFINITE_WINDOW  double  stdDev  
[BugLab_Wrong_Operator]^outBuffer.append ( "n: " + getN (   instanceof   )  + "\n" ) ;^280^^^^^277^289^outBuffer.append ( "n: " + getN (  )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "n: "  &&  getN (  )  + "\n" ) ;^280^^^^^277^289^outBuffer.append ( "n: " + getN (  )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "min: " + getMin (  !=  )  + "\n" ) ;^281^^^^^277^289^outBuffer.append ( "min: " + getMin (  )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "min: "  <  getMin (  )  + "\n" ) ;^281^^^^^277^289^outBuffer.append ( "min: " + getMin (  )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "max: " + getMax (  >=  )  + "\n" ) ;^282^^^^^277^289^outBuffer.append ( "max: " + getMax (  )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "max: "  &  getMax (  )  + "\n" ) ;^282^^^^^277^289^outBuffer.append ( "max: " + getMax (  )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "mean: " + getMean (  <<  )  + "\n" ) ;^283^^^^^277^289^outBuffer.append ( "mean: " + getMean (  )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "mean: "   instanceof   getMean (  )  + "\n" ) ;^283^^^^^277^289^outBuffer.append ( "mean: " + getMean (  )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "std dev: " + getStandardDeviation (  <=  )  + "\n" ) ;^284^^^^^277^289^outBuffer.append ( "std dev: " + getStandardDeviation (  )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "std dev: "  &  getStandardDeviation (  )  + "\n" ) ;^284^^^^^277^289^outBuffer.append ( "std dev: " + getStandardDeviation (  )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "median: " + getPercentile ( 50 )  <<  + "\n" ) ;^285^^^^^277^289^outBuffer.append ( "median: " + getPercentile ( 50 )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "median: "  &  getPercentile ( 50 )  + "\n" ) ;^285^^^^^277^289^outBuffer.append ( "median: " + getPercentile ( 50 )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Literal]^outBuffer.append ( "median: " + getPercentile ( index )  + "\n" ) ;^285^^^^^277^289^outBuffer.append ( "median: " + getPercentile ( 50 )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Literal]^outBuffer.append ( "median: " + getPercentile ( INFINITE_WINDOW )  + "\n" ) ;^285^^^^^277^289^outBuffer.append ( "median: " + getPercentile ( 50 )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "skewness: " + getSkewness (  <<  )  + "\n" ) ;^286^^^^^277^289^outBuffer.append ( "skewness: " + getSkewness (  )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "skewness: "   instanceof   getSkewness (  )  + "\n" ) ;^286^^^^^277^289^outBuffer.append ( "skewness: " + getSkewness (  )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "kurtosis: " + getKurtosis (  <  )  + "\n" ) ;^287^^^^^277^289^outBuffer.append ( "kurtosis: " + getKurtosis (  )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
[BugLab_Wrong_Operator]^outBuffer.append ( "kurtosis: "  ^  getKurtosis (  )  + "\n" ) ;^287^^^^^277^289^outBuffer.append ( "kurtosis: " + getKurtosis (  )  + "\n" ) ;^[CLASS] DescriptiveStatistics  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  outBuffer  boolean  long  serialVersionUID  int  INFINITE_WINDOW  index  windowSize  
