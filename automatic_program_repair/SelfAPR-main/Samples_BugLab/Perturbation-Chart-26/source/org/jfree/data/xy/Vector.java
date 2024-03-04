[BugLab_Variable_Misuse]^this.x = y;^70^^^^^69^72^this.x = x;^[CLASS] Vector  [METHOD] <init> [RETURN_TYPE] Vector(double,double)   double x double y [VARIABLES] double  x  y  boolean  
[BugLab_Variable_Misuse]^this.y = x;^71^^^^^69^72^this.y = y;^[CLASS] Vector  [METHOD] <init> [RETURN_TYPE] Vector(double,double)   double x double y [VARIABLES] double  x  y  boolean  
[BugLab_Variable_Misuse]^return y;^80^^^^^79^81^return this.x;^[CLASS] Vector  [METHOD] getX [RETURN_TYPE] double   [VARIABLES] double  x  y  boolean  
[BugLab_Variable_Misuse]^return y;^89^^^^^88^90^return this.y;^[CLASS] Vector  [METHOD] getY [RETURN_TYPE] double   [VARIABLES] double  x  y  boolean  
[BugLab_Variable_Misuse]^return Math.sqrt (  ( y * this.x )  +  ( this.y * this.y )  ) ;^98^^^^^97^99^return Math.sqrt (  ( this.x * this.x )  +  ( this.y * this.y )  ) ;^[CLASS] Vector  [METHOD] getLength [RETURN_TYPE] double   [VARIABLES] double  x  y  boolean  
[BugLab_Variable_Misuse]^return Math.sqrt (  ( this.x * this.x )  +  ( y * this.y )  ) ;^98^^^^^97^99^return Math.sqrt (  ( this.x * this.x )  +  ( this.y * this.y )  ) ;^[CLASS] Vector  [METHOD] getLength [RETURN_TYPE] double   [VARIABLES] double  x  y  boolean  
[BugLab_Argument_Swapping]^return Math.sqrt (  ( this.y * this.x )  +  ( this.x * this.y )  ) ;^98^^^^^97^99^return Math.sqrt (  ( this.x * this.x )  +  ( this.y * this.y )  ) ;^[CLASS] Vector  [METHOD] getLength [RETURN_TYPE] double   [VARIABLES] double  x  y  boolean  
[BugLab_Wrong_Operator]^return Math.sqrt (  &&  ( this.x * this.x )  +  ( this.y * this.y )  ) ;^98^^^^^97^99^return Math.sqrt (  ( this.x * this.x )  +  ( this.y * this.y )  ) ;^[CLASS] Vector  [METHOD] getLength [RETURN_TYPE] double   [VARIABLES] double  x  y  boolean  
[BugLab_Wrong_Operator]^return Math.sqrt (  ( this.x / this.x )  +  ( this.y * this.y )  ) ;^98^^^^^97^99^return Math.sqrt (  ( this.x * this.x )  +  ( this.y * this.y )  ) ;^[CLASS] Vector  [METHOD] getLength [RETURN_TYPE] double   [VARIABLES] double  x  y  boolean  
[BugLab_Wrong_Operator]^return Math.sqrt (  ( this.x * this.x )  +  ( this.y + this.y )  ) ;^98^^^^^97^99^return Math.sqrt (  ( this.x * this.x )  +  ( this.y * this.y )  ) ;^[CLASS] Vector  [METHOD] getLength [RETURN_TYPE] double   [VARIABLES] double  x  y  boolean  
[BugLab_Wrong_Operator]^return Math.sqrt (  ||  ( this.x * this.x )  +  ( this.y * this.y )  ) ;^98^^^^^97^99^return Math.sqrt (  ( this.x * this.x )  +  ( this.y * this.y )  ) ;^[CLASS] Vector  [METHOD] getLength [RETURN_TYPE] double   [VARIABLES] double  x  y  boolean  
[BugLab_Wrong_Operator]^return Math.sqrt (  ( this.x - this.x )  +  ( this.y * this.y )  ) ;^98^^^^^97^99^return Math.sqrt (  ( this.x * this.x )  +  ( this.y * this.y )  ) ;^[CLASS] Vector  [METHOD] getLength [RETURN_TYPE] double   [VARIABLES] double  x  y  boolean  
[BugLab_Variable_Misuse]^return Math.atan2 ( y, this.x ) ;^107^^^^^106^108^return Math.atan2 ( this.y, this.x ) ;^[CLASS] Vector  [METHOD] getAngle [RETURN_TYPE] double   [VARIABLES] double  x  y  boolean  
[BugLab_Variable_Misuse]^return Math.atan2 ( this.y, y ) ;^107^^^^^106^108^return Math.atan2 ( this.y, this.x ) ;^[CLASS] Vector  [METHOD] getAngle [RETURN_TYPE] double   [VARIABLES] double  x  y  boolean  
[BugLab_Argument_Swapping]^return Math.atan2 ( this.x, this.y ) ;^107^^^^^106^108^return Math.atan2 ( this.y, this.x ) ;^[CLASS] Vector  [METHOD] getAngle [RETURN_TYPE] double   [VARIABLES] double  x  y  boolean  
[BugLab_Wrong_Operator]^if  ( obj != this )  {^118^^^^^117^132^if  ( obj == this )  {^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Wrong_Literal]^return false;^119^^^^^117^132^return true;^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Wrong_Operator]^if  ( ! ( obj  >=  Vector )  )  {^121^^^^^117^132^if  ( ! ( obj instanceof Vector )  )  {^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Wrong_Literal]^return true;^122^^^^^117^132^return false;^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Variable_Misuse]^if  ( y != that.x )  {^125^^^^^117^132^if  ( this.x != that.x )  {^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Variable_Misuse]^if  ( this.x != y )  {^125^^^^^117^132^if  ( this.x != that.x )  {^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Argument_Swapping]^if  ( this.x != that.x.x )  {^125^^^^^117^132^if  ( this.x != that.x )  {^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Argument_Swapping]^if  ( that.x != this.x )  {^125^^^^^117^132^if  ( this.x != that.x )  {^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Wrong_Operator]^if  ( this.x == that.x )  {^125^^^^^117^132^if  ( this.x != that.x )  {^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Wrong_Literal]^return true;^126^^^^^117^132^return false;^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Variable_Misuse]^if  ( y != that.y )  {^128^^^^^117^132^if  ( this.y != that.y )  {^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Variable_Misuse]^if  ( this.y != y )  {^128^^^^^117^132^if  ( this.y != that.y )  {^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Argument_Swapping]^if  ( this.y != that.y.y )  {^128^^^^^117^132^if  ( this.y != that.y )  {^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Argument_Swapping]^if  ( that != this.y.y )  {^128^^^^^117^132^if  ( this.y != that.y )  {^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Argument_Swapping]^if  ( this.y != that )  {^128^^^^^117^132^if  ( this.y != that.y )  {^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Wrong_Operator]^if  ( this.y >= that.y )  {^128^^^^^117^132^if  ( this.y != that.y )  {^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Wrong_Literal]^return true;^129^^^^^117^132^return false;^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Wrong_Literal]^return false;^131^^^^^117^132^return true;^[CLASS] Vector  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  Vector  that  double  x  y  
[BugLab_Wrong_Literal]^int result = ;^140^^^^^139^146^int result = 193;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
[BugLab_Variable_Misuse]^long temp = Double.doubleToLongBits ( y ) ;^141^^^^^139^146^long temp = Double.doubleToLongBits ( this.x ) ;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
[BugLab_Argument_Swapping]^result = 37 * temp +  ( int )   ( result ^  ( temp >>> 32 )  ) ;^142^^^^^139^146^result = 37 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
[BugLab_Wrong_Operator]^result = 37 * result +  <=  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^142^^^^^139^146^result = 37 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
[BugLab_Wrong_Operator]^result = 37 + result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^142^^^^^139^146^result = 37 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
[BugLab_Wrong_Operator]^result = 37 * result +  ( int )   ( temp ^  ( temp  >  32 )  ) ;^142^^^^^139^146^result = 37 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
[BugLab_Wrong_Literal]^result = result * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^142^^^^^139^146^result = 37 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
[BugLab_Wrong_Literal]^result = 37 * result +  ( int )   ( temp ^  ( temp >>> result )  ) ;^142^^^^^139^146^result = 37 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
[BugLab_Variable_Misuse]^temp = Double.doubleToLongBits ( y ) ;^143^^^^^139^146^temp = Double.doubleToLongBits ( this.y ) ;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
[BugLab_Argument_Swapping]^result = 37 * temp +  ( int )   ( result ^  ( temp >>> 32 )  ) ;^144^^^^^139^146^result = 37 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
[BugLab_Wrong_Operator]^result = 37 * result +  ^  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^144^^^^^139^146^result = 37 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
[BugLab_Wrong_Operator]^result = 37 / result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^144^^^^^139^146^result = 37 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
[BugLab_Wrong_Operator]^result = 37 * result +  ( int )   ( temp ^  ( temp  ||  32 )  ) ;^144^^^^^139^146^result = 37 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
[BugLab_Wrong_Literal]^result = 36 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^144^^^^^139^146^result = 37 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
[BugLab_Wrong_Literal]^result = 37 * result +  ( int )   ( temp ^  ( temp >>> result )  ) ;^144^^^^^139^146^result = 37 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] Vector  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  double  x  y  int  result  long  temp  
