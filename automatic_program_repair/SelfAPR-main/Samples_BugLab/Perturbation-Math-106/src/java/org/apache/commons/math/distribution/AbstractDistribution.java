[BugLab_Argument_Swapping]^if  ( x1 > x0 )  {^59^^^^^57^64^if  ( x0 > x1 )  {^[CLASS] AbstractDistribution  [METHOD] cumulativeProbability [RETURN_TYPE] double   double x0 double x1 [VARIABLES] long  serialVersionUID  double  x0  x1  boolean  
[BugLab_Wrong_Operator]^if  ( x0 < x1 )  {^59^^^^^57^64^if  ( x0 > x1 )  {^[CLASS] AbstractDistribution  [METHOD] cumulativeProbability [RETURN_TYPE] double   double x0 double x1 [VARIABLES] long  serialVersionUID  double  x0  x1  boolean  
[BugLab_Argument_Swapping]^return cumulativeProbability ( x0 )  - cumulativeProbability ( x1 ) ;^63^^^^^57^64^return cumulativeProbability ( x1 )  - cumulativeProbability ( x0 ) ;^[CLASS] AbstractDistribution  [METHOD] cumulativeProbability [RETURN_TYPE] double   double x0 double x1 [VARIABLES] long  serialVersionUID  double  x0  x1  boolean  
[BugLab_Wrong_Operator]^return cumulativeProbability ( x1 )   >=  cumulativeProbability ( x0 ) ;^63^^^^^57^64^return cumulativeProbability ( x1 )  - cumulativeProbability ( x0 ) ;^[CLASS] AbstractDistribution  [METHOD] cumulativeProbability [RETURN_TYPE] double   double x0 double x1 [VARIABLES] long  serialVersionUID  double  x0  x1  boolean  
[BugLab_Variable_Misuse]^return cumulativeProbability ( x0 )  - cumulativeProbability ( x0 ) ;^63^^^^^57^64^return cumulativeProbability ( x1 )  - cumulativeProbability ( x0 ) ;^[CLASS] AbstractDistribution  [METHOD] cumulativeProbability [RETURN_TYPE] double   double x0 double x1 [VARIABLES] long  serialVersionUID  double  x0  x1  boolean  
[BugLab_Variable_Misuse]^return cumulativeProbability ( x1 )  - cumulativeProbability ( x1 ) ;^63^^^^^57^64^return cumulativeProbability ( x1 )  - cumulativeProbability ( x0 ) ;^[CLASS] AbstractDistribution  [METHOD] cumulativeProbability [RETURN_TYPE] double   double x0 double x1 [VARIABLES] long  serialVersionUID  double  x0  x1  boolean  
