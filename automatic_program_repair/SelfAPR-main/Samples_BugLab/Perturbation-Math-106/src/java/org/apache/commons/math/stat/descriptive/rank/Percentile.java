[BugLab_Variable_Misuse]^return evaluate ( values, 0, values.length, quantile ) ;^120^^^^^118^121^return evaluate ( values, 0, values.length, p ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final double p [VARIABLES] boolean  double[]  values  double  p  quantile  long  serialVersionUID  
[BugLab_Argument_Swapping]^return evaluate ( values.length, 0, values, p ) ;^120^^^^^118^121^return evaluate ( values, 0, values.length, p ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final double p [VARIABLES] boolean  double[]  values  double  p  quantile  long  serialVersionUID  
[BugLab_Argument_Swapping]^return evaluate ( values, 0, p, values.length ) ;^120^^^^^118^121^return evaluate ( values, 0, values.length, p ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final double p [VARIABLES] boolean  double[]  values  double  p  quantile  long  serialVersionUID  
[BugLab_Wrong_Literal]^return evaluate ( values, 1, values.length, p ) ;^120^^^^^118^121^return evaluate ( values, 0, values.length, p ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final double p [VARIABLES] boolean  double[]  values  double  p  quantile  long  serialVersionUID  
[BugLab_Wrong_Literal]^return evaluate ( values, -1, values.length, p ) ;^120^^^^^118^121^return evaluate ( values, 0, values.length, p ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final double p [VARIABLES] boolean  double[]  values  double  p  quantile  long  serialVersionUID  
[BugLab_Variable_Misuse]^return evaluate ( values, start, start, quantile ) ;^148^^^^^147^149^return evaluate ( values, start, length, quantile ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int start final int length [VARIABLES] boolean  double[]  values  double  p  quantile  long  serialVersionUID  int  length  start  
[BugLab_Variable_Misuse]^return evaluate ( values, start, length, p ) ;^148^^^^^147^149^return evaluate ( values, start, length, quantile ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int start final int length [VARIABLES] boolean  double[]  values  double  p  quantile  long  serialVersionUID  int  length  start  
[BugLab_Argument_Swapping]^return evaluate ( length, start, values, quantile ) ;^148^^^^^147^149^return evaluate ( values, start, length, quantile ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int start final int length [VARIABLES] boolean  double[]  values  double  p  quantile  long  serialVersionUID  int  length  start  
[BugLab_Argument_Swapping]^return evaluate ( values, length, start, quantile ) ;^148^^^^^147^149^return evaluate ( values, start, length, quantile ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int start final int length [VARIABLES] boolean  double[]  values  double  p  quantile  long  serialVersionUID  int  length  start  
[BugLab_Argument_Swapping]^return evaluate ( values, quantile, length, start ) ;^148^^^^^147^149^return evaluate ( values, start, length, quantile ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int start final int length [VARIABLES] boolean  double[]  values  double  p  quantile  long  serialVersionUID  int  length  start  
[BugLab_Argument_Swapping]^return evaluate ( start, values, length, quantile ) ;^148^^^^^147^149^return evaluate ( values, start, length, quantile ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int start final int length [VARIABLES] boolean  double[]  values  double  p  quantile  long  serialVersionUID  int  length  start  
[BugLab_Argument_Swapping]^return evaluate ( values, start, quantile, length ) ;^148^^^^^147^149^return evaluate ( values, start, length, quantile ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int start final int length [VARIABLES] boolean  double[]  values  double  p  quantile  long  serialVersionUID  int  length  start  
[BugLab_Variable_Misuse]^test ( values, intPos, length ) ;^184^^^^^169^199^test ( values, begin, length ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^test ( values, begin, intPos ) ;^184^^^^^169^199^test ( values, begin, length ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Argument_Swapping]^test ( length, begin, values ) ;^184^^^^^169^199^test ( values, begin, length ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Argument_Swapping]^test ( values, length, begin ) ;^184^^^^^169^199^test ( values, begin, length ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^if  (  ( p > 100 )  &&  ( p <= 0 )  )  {^186^^^^^171^201^if  (  ( p > 100 )  ||  ( p <= 0 )  )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^if  (  ( p >= 100 )  ||  ( p <= 0 )  )  {^186^^^^^171^201^if  (  ( p > 100 )  ||  ( p <= 0 )  )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^if  (  ( p > 100 )  ||  ( p == 0 )  )  {^186^^^^^171^201^if  (  ( p > 100 )  ||  ( p <= 0 )  )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Literal]^if  (  ( p >  )  ||  ( p <= 0 )  )  {^186^^^^^171^201^if  (  ( p > 100 )  ||  ( p <= 0 )  )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Literal]^if  (  ( p > 1intPosintPos )  ||  ( p <= intPos )  )  {^186^^^^^171^201^if  (  ( p > 100 )  ||  ( p <= 0 )  )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^if  ( length != 0 )  {^189^^^^^174^204^if  ( length == 0 )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Literal]^if  ( length == intPos )  {^189^^^^^174^204^if  ( length == 0 )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^if  ( intPos == 1 )  {^192^^^^^177^207^if  ( length == 1 )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^if  ( length < 1 )  {^192^^^^^177^207^if  ( length == 1 )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Literal]^if  ( length == 0 )  {^192^^^^^177^207^if  ( length == 1 )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^return sorted[begin];^193^^^^^178^208^return values[begin];^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^double upperos = p *  ( n + 1 )  / 100;^196^^^^^181^211^double pos = p *  ( n + 1 )  / 100;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^double pos = p *  ( upper + 1 )  / 100;^196^^^^^181^211^double pos = p *  ( n + 1 )  / 100;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Argument_Swapping]^double nos = p *  ( p + 1 )  / 100;^196^^^^^181^211^double pos = p *  ( n + 1 )  / 100;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^double pos = p *  ( n + 1 )  * 100;^196^^^^^181^211^double pos = p *  ( n + 1 )  / 100;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^double / pos = p *  ( n + 1 )  / 100;^196^^^^^181^211^double pos = p *  ( n + 1 )  / 100;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^double pos = p *  ( n  <<  1 )  / 100;^196^^^^^181^211^double pos = p *  ( n + 1 )  / 100;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Literal]^double pos = p *  ( n  )  / 100;^196^^^^^181^211^double pos = p *  ( n + 1 )  / 100;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Literal]^double pos = p *  ( n + 1 )  / begin;^196^^^^^181^211^double pos = p *  ( n + 1 )  / 100;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^double fpos = Math.floor ( p ) ;^197^^^^^182^212^double fpos = Math.floor ( pos ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^double dif = upper - fpos;^199^^^^^184^214^double dif = pos - fpos;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^double dif = pos - quantile;^199^^^^^184^214^double dif = pos - fpos;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Argument_Swapping]^double dif = fpos - pos;^199^^^^^184^214^double dif = pos - fpos;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^double dif = pos  &  fpos;^199^^^^^184^214^double dif = pos - fpos;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^System.arraycopy ( sorted, begin, sorted, 0, length ) ;^201^^^^^186^216^System.arraycopy ( values, begin, sorted, 0, length ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Argument_Swapping]^System.arraycopy ( begin, values, sorted, 0, length ) ;^201^^^^^186^216^System.arraycopy ( values, begin, sorted, 0, length ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Argument_Swapping]^System.arraycopy ( values, length, sorted, 0, begin ) ;^201^^^^^186^216^System.arraycopy ( values, begin, sorted, 0, length ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Argument_Swapping]^System.arraycopy ( sorted, begin, values, 0, length ) ;^201^^^^^186^216^System.arraycopy ( values, begin, sorted, 0, length ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Literal]^System.arraycopy ( values, begin, sorted, , length ) ;^201^^^^^186^216^System.arraycopy ( values, begin, sorted, 0, length ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^Arrays.sort ( values ) ;^202^^^^^187^217^Arrays.sort ( sorted ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^if  ( upper < 1 )  {^204^^^^^189^219^if  ( pos < 1 )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^if  ( pos <= 1 )  {^204^^^^^189^219^if  ( pos < 1 )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^return values[0];^205^^^^^190^220^return sorted[0];^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Literal]^return sorted[intPos];^205^^^^^190^220^return sorted[0];^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^if  ( upper >= n )  {^207^^^^^192^222^if  ( pos >= n )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^if  ( pos >= upper )  {^207^^^^^192^222^if  ( pos >= n )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Argument_Swapping]^if  ( n >= pos )  {^207^^^^^192^222^if  ( pos >= n )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^if  ( pos == n )  {^207^^^^^192^222^if  ( pos >= n )  {^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^return values[length - 1];^208^^^^^193^223^return sorted[length - 1];^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^return sorted[intPos - 1];^208^^^^^193^223^return sorted[length - 1];^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^return sorted[length  |  1];^208^^^^^193^223^return sorted[length - 1];^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Literal]^return sorted[length - ];^208^^^^^193^223^return sorted[length - 1];^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^return sorted[length  <=  1];^208^^^^^193^223^return sorted[length - 1];^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Literal]^return sorted[length - begin];^208^^^^^193^223^return sorted[length - 1];^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^double lower = values[intPos - 1];^210^^^^^195^225^double lower = sorted[intPos - 1];^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^double lower = sorted[length - 1];^210^^^^^195^225^double lower = sorted[intPos - 1];^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^double lower = sorted[intPos  <=  1];^210^^^^^195^225^double lower = sorted[intPos - 1];^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Literal]^double lower = sorted[intPos - intPos];^210^^^^^195^225^double lower = sorted[intPos - 1];^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^return quantile + dif *  ( upper - lower ) ;^212^^^^^197^227^return lower + dif *  ( upper - lower ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^return lower + upper *  ( upper - lower ) ;^212^^^^^197^227^return lower + dif *  ( upper - lower ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^return lower + dif *  ( quantile - lower ) ;^212^^^^^197^227^return lower + dif *  ( upper - lower ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Argument_Swapping]^return dif + lower *  ( upper - lower ) ;^212^^^^^197^227^return lower + dif *  ( upper - lower ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Argument_Swapping]^return lower + upper *  ( dif - lower ) ;^212^^^^^197^227^return lower + dif *  ( upper - lower ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^return lower + dif *  >  ( upper - lower ) ;^212^^^^^197^227^return lower + dif *  ( upper - lower ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^return + lower + dif *  ( upper - lower ) ;^212^^^^^197^227^return lower + dif *  ( upper - lower ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Wrong_Operator]^return lower + dif *  ( upper  ^  lower ) ;^212^^^^^197^227^return lower + dif *  ( upper - lower ) ;^[CLASS] Percentile  [METHOD] evaluate [RETURN_TYPE] double   final double[] values final int begin final int length final double p [VARIABLES] boolean  double[]  sorted  values  double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  int  begin  intPos  length  
[BugLab_Variable_Misuse]^return upper;^222^^^^^221^223^return quantile;^[CLASS] Percentile  [METHOD] getQuantile [RETURN_TYPE] double   [VARIABLES] double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( p <= 0 && p > 100 )  {^234^^^^^233^238^if  ( p <= 0 || p > 100 )  {^[CLASS] Percentile  [METHOD] setQuantile [RETURN_TYPE] void   final double p [VARIABLES] double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( p == 0 || p > 100 )  {^234^^^^^233^238^if  ( p <= 0 || p > 100 )  {^[CLASS] Percentile  [METHOD] setQuantile [RETURN_TYPE] void   final double p [VARIABLES] double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( p <= 0 || p >= 100 )  {^234^^^^^233^238^if  ( p <= 0 || p > 100 )  {^[CLASS] Percentile  [METHOD] setQuantile [RETURN_TYPE] void   final double p [VARIABLES] double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  boolean  
[BugLab_Wrong_Literal]^if  ( p <= 0 || p > 99 )  {^234^^^^^233^238^if  ( p <= 0 || p > 100 )  {^[CLASS] Percentile  [METHOD] setQuantile [RETURN_TYPE] void   final double p [VARIABLES] double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^quantile = pos;^237^^^^^233^238^quantile = p;^[CLASS] Percentile  [METHOD] setQuantile [RETURN_TYPE] void   final double p [VARIABLES] double  dif  fpos  lower  n  p  pos  quantile  upper  long  serialVersionUID  boolean  
