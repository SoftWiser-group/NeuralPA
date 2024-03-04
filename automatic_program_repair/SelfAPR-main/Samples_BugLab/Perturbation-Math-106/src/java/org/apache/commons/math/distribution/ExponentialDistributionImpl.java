[BugLab_Wrong_Operator]^if  ( mean > 0.0 )  {^51^^^^^50^55^if  ( mean <= 0.0 )  {^[CLASS] ExponentialDistributionImpl  [METHOD] setMean [RETURN_TYPE] void   double mean [VARIABLES] double  mean  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^if  ( ret <= 0.0 )  {^82^^^^^80^88^if  ( x <= 0.0 )  {^[CLASS] ExponentialDistributionImpl  [METHOD] cumulativeProbability [RETURN_TYPE] double   double x [VARIABLES] double  mean  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( x < 0.0 )  {^82^^^^^80^88^if  ( x <= 0.0 )  {^[CLASS] ExponentialDistributionImpl  [METHOD] cumulativeProbability [RETURN_TYPE] double   double x [VARIABLES] double  mean  ret  x  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^ret = 1.0 - Math.exp ( -ret / getMean (  )  ) ;^85^^^^^80^88^ret = 1.0 - Math.exp ( -x / getMean (  )  ) ;^[CLASS] ExponentialDistributionImpl  [METHOD] cumulativeProbability [RETURN_TYPE] double   double x [VARIABLES] double  mean  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^ret = 1.0  <  Math.exp ( -x / getMean (  )  ) ;^85^^^^^80^88^ret = 1.0 - Math.exp ( -x / getMean (  )  ) ;^[CLASS] ExponentialDistributionImpl  [METHOD] cumulativeProbability [RETURN_TYPE] double   double x [VARIABLES] double  mean  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^ret = 1.0 - Math.exp ( -x + getMean (  )  ) ;^85^^^^^80^88^ret = 1.0 - Math.exp ( -x / getMean (  )  ) ;^[CLASS] ExponentialDistributionImpl  [METHOD] cumulativeProbability [RETURN_TYPE] double   double x [VARIABLES] double  mean  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^ret = 1.0 - Math.exp ( -x - getMean (  )  ) ;^85^^^^^80^88^ret = 1.0 - Math.exp ( -x / getMean (  )  ) ;^[CLASS] ExponentialDistributionImpl  [METHOD] cumulativeProbability [RETURN_TYPE] double   double x [VARIABLES] double  mean  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^ret = 1.0 - Math.exp ( -x * getMean (  )  ) ;^85^^^^^80^88^ret = 1.0 - Math.exp ( -x / getMean (  )  ) ;^[CLASS] ExponentialDistributionImpl  [METHOD] cumulativeProbability [RETURN_TYPE] double   double x [VARIABLES] double  mean  ret  x  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return x;^87^^^^^80^88^return ret;^[CLASS] ExponentialDistributionImpl  [METHOD] cumulativeProbability [RETURN_TYPE] double   double x [VARIABLES] double  mean  ret  x  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^if  ( x < 0.0 || p > 1.0 )  {^105^^^^^102^115^if  ( p < 0.0 || p > 1.0 )  {^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( p < 0.0 && p > 1.0 )  {^105^^^^^102^115^if  ( p < 0.0 || p > 1.0 )  {^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( p <= 0.0 || p > 1.0 )  {^105^^^^^102^115^if  ( p < 0.0 || p > 1.0 )  {^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( p < 0.0 || p >= 1.0 )  {^105^^^^^102^115^if  ( p < 0.0 || p > 1.0 )  {^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^} else if  ( p >= 1.0 )  {^108^^^^^102^115^} else if  ( p == 1.0 )  {^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^ret - = -getMean (  )  * Math.log ( 1.0 - p ) ;^111^^^^^102^115^ret = -getMean (  )  * Math.log ( 1.0 - p ) ;^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^ret = -getMean (  )  * Math.log ( 1.0  >  p ) ;^111^^^^^102^115^ret = -getMean (  )  * Math.log ( 1.0 - p ) ;^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^ret = -getMean (  )  * Math.log ( 1.0 - x ) ;^111^^^^^102^115^ret = -getMean (  )  * Math.log ( 1.0 - p ) ;^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^ret = -getMean (  )  * Math.log ( 1.0  <<  p ) ;^111^^^^^102^115^ret = -getMean (  )  * Math.log ( 1.0 - p ) ;^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^ret / = -getMean (  )  * Math.log ( 1.0 - p ) ;^111^^^^^102^115^ret = -getMean (  )  * Math.log ( 1.0 - p ) ;^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^ret = -getMean (  )  * Math.log ( 1.0   instanceof   p ) ;^111^^^^^102^115^ret = -getMean (  )  * Math.log ( 1.0 - p ) ;^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^} else if  ( x == 1.0 )  {^108^^^^^102^115^} else if  ( p == 1.0 )  {^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^} else if  ( p != 1.0 )  {^108^^^^^102^115^} else if  ( p == 1.0 )  {^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^ret + = -getMean (  )  * Math.log ( 1.0 - p ) ;^111^^^^^102^115^ret = -getMean (  )  * Math.log ( 1.0 - p ) ;^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^ret = -getMean (  )  * Math.log ( 1.0  <=  p ) ;^111^^^^^102^115^ret = -getMean (  )  * Math.log ( 1.0 - p ) ;^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^ret = -getMean (  )  * Math.log ( 1.0  ||  p ) ;^111^^^^^102^115^ret = -getMean (  )  * Math.log ( 1.0 - p ) ;^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return x;^114^^^^^102^115^return ret;^[CLASS] ExponentialDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Literal]^return -1;^126^^^^^125^127^return 0;^[CLASS] ExponentialDistributionImpl  [METHOD] getDomainLowerBound [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^if  ( x < .5 )  {^141^^^^^137^148^if  ( p < .5 )  {^[CLASS] ExponentialDistributionImpl  [METHOD] getDomainUpperBound [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( p == .5 )  {^141^^^^^137^148^if  ( p < .5 )  {^[CLASS] ExponentialDistributionImpl  [METHOD] getDomainUpperBound [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^if  ( x < .5 )  {^160^^^^^157^167^if  ( p < .5 )  {^[CLASS] ExponentialDistributionImpl  [METHOD] getInitialDomain [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( p <= .5 )  {^160^^^^^157^167^if  ( p < .5 )  {^[CLASS] ExponentialDistributionImpl  [METHOD] getInitialDomain [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^return + getMean (  )  * .5;^162^^^^^157^167^return getMean (  )  * .5;^[CLASS] ExponentialDistributionImpl  [METHOD] getInitialDomain [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^return - getMean (  )  * .5;^162^^^^^157^167^return getMean (  )  * .5;^[CLASS] ExponentialDistributionImpl  [METHOD] getInitialDomain [RETURN_TYPE] double   double p [VARIABLES] double  mean  p  ret  x  long  serialVersionUID  boolean  
