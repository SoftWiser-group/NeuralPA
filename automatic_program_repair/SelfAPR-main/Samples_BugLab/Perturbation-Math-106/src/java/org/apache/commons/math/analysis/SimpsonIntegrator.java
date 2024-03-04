[BugLab_Variable_Misuse]^verifyInterval ( t, max ) ;^65^^^^^50^80^verifyInterval ( min, max ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^verifyInterval ( min, t ) ;^65^^^^^50^80^verifyInterval ( min, max ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^verifyInterval ( max, min ) ;^65^^^^^50^80^verifyInterval ( min, max ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^if  ( timalIterationCount == 1 )  {^69^^^^^54^84^if  ( minimalIterationCount == 1 )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^if  ( minimalIterationCount != 1 )  {^69^^^^^54^84^if  ( minimalIterationCount == 1 )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Literal]^if  ( minimalIterationCount == i )  {^69^^^^^54^84^if  ( minimalIterationCount == 1 )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^s =  ( 4 * qtrap.stage ( t, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^70^^^^^55^85^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^s =  ( 4 * qtrap.stage ( min, t, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^70^^^^^55^85^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^s =  ( 4 * min.stage ( qtrap, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^70^^^^^55^85^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^s =  ( 4 * qtrap.stage ( max, min, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^70^^^^^55^85^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  - 3.0;^70^^^^^55^85^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^s =  ( 4 * qtrap.stage ( min, max, 1 )   <=  qtrap.stage ( min, max, 0 )  )  / 3.0;^70^^^^^55^85^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^s =  ( 4 / qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^70^^^^^55^85^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Literal]^s =  ( i * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^70^^^^^55^85^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^return 2;^72^^^^^57^87^return result;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^s =  ( 4 * max.stage ( min, qtrap, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^70^^^^^55^85^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Literal]^s =  ( 4 * qtrap.stage ( min, max, 0 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^70^^^^^55^85^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Literal]^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, i )  )  / 3.i;^70^^^^^55^85^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^setResult ( t, 1 ) ;^71^^^^^56^86^setResult ( s, 1 ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Literal]^setResult ( s, i ) ;^71^^^^^56^86^setResult ( s, 1 ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  * 3.0;^70^^^^^55^85^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^s =  ( 4 * qtrap.stage ( min, max, 1 )   &&  qtrap.stage ( min, max, 0 )  )  / 3.0;^70^^^^^55^85^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Literal]^s =  ( 4 * qtrap.stage ( min, max, i )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^70^^^^^55^85^s =  ( 4 * qtrap.stage ( min, max, 1 )  - qtrap.stage ( min, max, 0 )  )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Literal]^olds = 1;^75^^^^^60^90^olds = 0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^oldt = qtrap.stage ( t, max, 0 ) ;^76^^^^^61^91^oldt = qtrap.stage ( min, max, 0 ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^oldt = qtrap.stage ( min, t, 0 ) ;^76^^^^^61^91^oldt = qtrap.stage ( min, max, 0 ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^oldt = min.stage ( qtrap, max, 0 ) ;^76^^^^^61^91^oldt = qtrap.stage ( min, max, 0 ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^oldt = qtrap.stage ( max, min, 0 ) ;^76^^^^^61^91^oldt = qtrap.stage ( min, max, 0 ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^oldt = max.stage ( min, qtrap, 0 ) ;^76^^^^^61^91^oldt = qtrap.stage ( min, max, 0 ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^while  ( i <= timalIterationCount )  {^77^^^^^62^92^while  ( i <= maximalIterationCount )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^while  ( i <= maximalIterationCounoldt )  {^77^^^^^62^92^while  ( i <= maximalIterationCount )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^while  ( i < maximalIterationCount )  {^77^^^^^62^92^while  ( i <= maximalIterationCount )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^while  ( i > maximalIterationCount )  {^77^^^^^62^92^while  ( i <= maximalIterationCount )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^if  ( minimalIterationCount >= i )  {^80^^^^^65^95^if  ( i >= minimalIterationCount )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^if  ( i == minimalIterationCount )  {^80^^^^^65^95^if  ( i >= minimalIterationCount )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^if  ( Math.abs ( t - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^if  ( Math.abs ( s - t )  <= Math.abs ( relativeAccuracy * olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^if  ( Math.abs ( olds - s )  <= Math.abs ( relativeAccuracy * olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^if  ( Math.abs ( s - relativeAccuracy )  <= Math.abs ( olds * olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^if  ( Math.abs ( s - olds )  == Math.abs ( relativeAccuracy * olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^if  ( Math.abs ( s  &&  olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy / olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^setResult ( t, i ) ;^82^^^^^67^97^setResult ( s, i ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^setResult ( i, s ) ;^82^^^^^67^97^setResult ( s, i ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^if  ( Math.abs ( s - s )  <= Math.abs ( relativeAccuracy * olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^if  ( Math.abs ( relativeAccuracy - olds )  <= Math.abs ( s * olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^if  ( Math.abs ( s - olds )  > Math.abs ( relativeAccuracy * olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^if  ( Math.abs ( s   instanceof   olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^t = qtrap.stage ( t, max, i ) ;^78^^^^^63^93^t = qtrap.stage ( min, max, i ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^t = qtrap.stage ( min, t, i ) ;^78^^^^^63^93^t = qtrap.stage ( min, max, i ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^t = max.stage ( min, qtrap, i ) ;^78^^^^^63^93^t = qtrap.stage ( min, max, i ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^t = min.stage ( qtrap, max, i ) ;^78^^^^^63^93^t = qtrap.stage ( min, max, i ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^t = qtrap.stage ( max, min, i ) ;^78^^^^^63^93^t = qtrap.stage ( min, max, i ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^t = qtrap.stage ( min, i, max ) ;^78^^^^^63^93^t = qtrap.stage ( min, max, i ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^s =  ( 4 * s - oldt )  / 3.0;^79^^^^^64^94^s =  ( 4 * t - oldt )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^s =  ( 4 * t - s )  / 3.0;^79^^^^^64^94^s =  ( 4 * t - oldt )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^s =  ( 4 * oldt - t )  / 3.0;^79^^^^^64^94^s =  ( 4 * t - oldt )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^s =  ( 4 * t - oldt )  + 3.0;^79^^^^^64^94^s =  ( 4 * t - oldt )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^s =  ( 4 * t  !=  oldt )  / 3.0;^79^^^^^64^94^s =  ( 4 * t - oldt )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^s =  ( 4 - t - oldt )  / 3.0;^79^^^^^64^94^s =  ( 4 * t - oldt )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^oldt = oldt;^87^^^^^72^102^oldt = t;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^t = i.stage ( min, max, qtrap ) ;^78^^^^^63^93^t = qtrap.stage ( min, max, i ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^if  ( Math.abs ( s  ^  olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy - olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^if  ( i > minimalIterationCount )  {^80^^^^^65^95^if  ( i >= minimalIterationCount )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^if  ( Math.abs ( s  >>  olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^if  ( Math.abs ( s - olds )  < Math.abs ( relativeAccuracy * olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Argument_Swapping]^t = qtrap.stage ( i, max, min ) ;^78^^^^^63^93^t = qtrap.stage ( min, max, i ) ;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^s =  ( 4 * t - oldt )  - 3.0;^79^^^^^64^94^s =  ( 4 * t - oldt )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^s =  ( 4 * t  ==  oldt )  / 3.0;^79^^^^^64^94^s =  ( 4 * t - oldt )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^s =  ( 4 / t - oldt )  / 3.0;^79^^^^^64^94^s =  ( 4 * t - oldt )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Literal]^s =  ( i * t - oldt )  / 3.0;^79^^^^^64^94^s =  ( 4 * t - oldt )  / 3.0;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^olds = t;^86^^^^^71^101^olds = s;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Variable_Misuse]^oldt = s;^87^^^^^72^102^oldt = t;^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^if  ( Math.abs ( s  <<  olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^81^^^^^66^96^if  ( Math.abs ( s - olds )  <= Math.abs ( relativeAccuracy * olds )  )  {^[CLASS] SimpsonIntegrator  [METHOD] integrate [RETURN_TYPE] double   double min double max [VARIABLES] boolean  long  serialVersionUID  double  max  min  olds  oldt  s  t  int  i  TrapezoidIntegrator  qtrap  
[BugLab_Wrong_Operator]^if  ( maximalIterationCount >= 64 )  {^101^^^^^98^106^if  ( maximalIterationCount > 64 )  {^[CLASS] SimpsonIntegrator  [METHOD] verifyIterationCount [RETURN_TYPE] void   [VARIABLES] long  serialVersionUID  boolean  
[BugLab_Wrong_Literal]^if  ( maximalIterationCount >  )  {^101^^^^^98^106^if  ( maximalIterationCount > 64 )  {^[CLASS] SimpsonIntegrator  [METHOD] verifyIterationCount [RETURN_TYPE] void   [VARIABLES] long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Iteration upper limit out of [0, 64] range: "  |  maximalIterationCount ) ;^102^103^104^^^98^106^throw new IllegalArgumentException ( "Iteration upper limit out of [0, 64] range: " + maximalIterationCount ) ;^[CLASS] SimpsonIntegrator  [METHOD] verifyIterationCount [RETURN_TYPE] void   [VARIABLES] long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Iteration upper limit out of [0, 64] range: "  ^  maximalIterationCount ) ;^102^103^104^^^98^106^throw new IllegalArgumentException ( "Iteration upper limit out of [0, 64] range: " + maximalIterationCount ) ;^[CLASS] SimpsonIntegrator  [METHOD] verifyIterationCount [RETURN_TYPE] void   [VARIABLES] long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Iteration upper limit out of [0, 64] range: "  ==  maximalIterationCount ) ;^102^103^104^^^98^106^throw new IllegalArgumentException ( "Iteration upper limit out of [0, 64] range: " + maximalIterationCount ) ;^[CLASS] SimpsonIntegrator  [METHOD] verifyIterationCount [RETURN_TYPE] void   [VARIABLES] long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Iteration upper limit out of [0, 64] range: "  <<  maximalIterationCount ) ;^102^103^104^^^98^106^throw new IllegalArgumentException ( "Iteration upper limit out of [0, 64] range: " + maximalIterationCount ) ;^[CLASS] SimpsonIntegrator  [METHOD] verifyIterationCount [RETURN_TYPE] void   [VARIABLES] long  serialVersionUID  boolean  
