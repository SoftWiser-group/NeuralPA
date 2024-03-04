[BugLab_Wrong_Operator]^setGamma ( DistributionFactory.newInstance (  ) .createGammaDistribution ( degreesOfFreedom - 2.0, 2.0 )  ) ;^43^44^^^^41^45^setGamma ( DistributionFactory.newInstance (  ) .createGammaDistribution ( degreesOfFreedom / 2.0, 2.0 )  ) ;^[CLASS] ChiSquaredDistributionImpl  [METHOD] <init> [RETURN_TYPE] ChiSquaredDistributionImpl(double)   double degreesOfFreedom [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  degreesOfFreedom  
[BugLab_Wrong_Operator]^setGamma ( DistributionFactory.newInstance (  ) .createGammaDistribution ( degreesOfFreedom + 2.0, 2.0 )  ) ;^43^44^^^^41^45^setGamma ( DistributionFactory.newInstance (  ) .createGammaDistribution ( degreesOfFreedom / 2.0, 2.0 )  ) ;^[CLASS] ChiSquaredDistributionImpl  [METHOD] <init> [RETURN_TYPE] ChiSquaredDistributionImpl(double)   double degreesOfFreedom [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  degreesOfFreedom  
[BugLab_Wrong_Operator]^getGamma (  ) .setAlpha ( degreesOfFreedom * 2.0 ) ;^52^^^^^51^53^getGamma (  ) .setAlpha ( degreesOfFreedom / 2.0 ) ;^[CLASS] ChiSquaredDistributionImpl  [METHOD] setDegreesOfFreedom [RETURN_TYPE] void   double degreesOfFreedom [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  degreesOfFreedom  
[BugLab_Wrong_Operator]^return / getGamma (  ) .getAlpha (  )  * 2.0;^60^^^^^59^61^return getGamma (  ) .getAlpha (  )  * 2.0;^[CLASS] ChiSquaredDistributionImpl  [METHOD] getDegreesOfFreedom [RETURN_TYPE] double   [VARIABLES] long  serialVersionUID  GammaDistribution  gamma  boolean  
[BugLab_Wrong_Operator]^if  ( p != 0 )  {^89^^^^^87^96^if  ( p == 0 )  {^[CLASS] ChiSquaredDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   final double p [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  p  
[BugLab_Wrong_Operator]^if  ( p >= 1 )  {^92^^^^^87^96^if  ( p == 1 )  {^[CLASS] ChiSquaredDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   final double p [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  p  
[BugLab_Wrong_Literal]^if  ( p == 2 )  {^92^^^^^87^96^if  ( p == 1 )  {^[CLASS] ChiSquaredDistributionImpl  [METHOD] inverseCumulativeProbability [RETURN_TYPE] double   final double p [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  p  
[BugLab_Variable_Misuse]^return Double.p * getGamma (  ) .getBeta (  ) ;^108^^^^^107^109^return Double.MIN_VALUE * getGamma (  ) .getBeta (  ) ;^[CLASS] ChiSquaredDistributionImpl  [METHOD] getDomainLowerBound [RETURN_TYPE] double   double p [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  p  
[BugLab_Wrong_Operator]^return - Double.MIN_VALUE * getGamma (  ) .getBeta (  ) ;^108^^^^^107^109^return Double.MIN_VALUE * getGamma (  ) .getBeta (  ) ;^[CLASS] ChiSquaredDistributionImpl  [METHOD] getDomainLowerBound [RETURN_TYPE] double   double p [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  p  
[BugLab_Variable_Misuse]^if  ( ret < .5 )  {^126^^^^^120^135^if  ( p < .5 )  {^[CLASS] ChiSquaredDistributionImpl  [METHOD] getDomainUpperBound [RETURN_TYPE] double   double p [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  p  ret  
[BugLab_Wrong_Operator]^if  ( p > .5 )  {^126^^^^^120^135^if  ( p < .5 )  {^[CLASS] ChiSquaredDistributionImpl  [METHOD] getDomainUpperBound [RETURN_TYPE] double   double p [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  p  ret  
[BugLab_Variable_Misuse]^return p;^134^^^^^120^135^return ret;^[CLASS] ChiSquaredDistributionImpl  [METHOD] getDomainUpperBound [RETURN_TYPE] double   double p [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  p  ret  
[BugLab_Variable_Misuse]^if  ( ret < .5 )  {^151^^^^^145^160^if  ( p < .5 )  {^[CLASS] ChiSquaredDistributionImpl  [METHOD] getInitialDomain [RETURN_TYPE] double   double p [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  p  ret  
[BugLab_Wrong_Operator]^if  ( p <= .5 )  {^151^^^^^145^160^if  ( p < .5 )  {^[CLASS] ChiSquaredDistributionImpl  [METHOD] getInitialDomain [RETURN_TYPE] double   double p [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  p  ret  
[BugLab_Wrong_Operator]^ret - = getDegreesOfFreedom (  )  * .5;^153^^^^^145^160^ret = getDegreesOfFreedom (  )  * .5;^[CLASS] ChiSquaredDistributionImpl  [METHOD] getInitialDomain [RETURN_TYPE] double   double p [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  p  ret  
[BugLab_Variable_Misuse]^return p;^159^^^^^145^160^return ret;^[CLASS] ChiSquaredDistributionImpl  [METHOD] getInitialDomain [RETURN_TYPE] double   double p [VARIABLES] boolean  long  serialVersionUID  GammaDistribution  gamma  double  p  ret  
