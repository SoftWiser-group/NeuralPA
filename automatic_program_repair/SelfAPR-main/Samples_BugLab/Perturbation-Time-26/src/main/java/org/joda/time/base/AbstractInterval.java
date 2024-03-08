[BugLab_Argument_Swapping]^if  ( start < end )  {^62^^^^^61^65^if  ( end < start )  {^[CLASS] AbstractInterval  [METHOD] checkInterval [RETURN_TYPE] void   long start long end [VARIABLES] boolean  long  end  start  
[BugLab_Wrong_Operator]^if  ( end <= start )  {^62^^^^^61^65^if  ( end < start )  {^[CLASS] AbstractInterval  [METHOD] checkInterval [RETURN_TYPE] void   long start long end [VARIABLES] boolean  long  end  start  
[BugLab_Variable_Misuse]^return  ( thisEnd >= thisStart && millisInstant < thisEnd ) ;^100^^^^^97^101^return  ( millisInstant >= thisStart && millisInstant < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   long millisInstant [VARIABLES] boolean  long  millisInstant  thisEnd  thisStart  
[BugLab_Variable_Misuse]^return  ( millisInstant >= thisStart && millisInstant < millisInstant ) ;^100^^^^^97^101^return  ( millisInstant >= thisStart && millisInstant < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   long millisInstant [VARIABLES] boolean  long  millisInstant  thisEnd  thisStart  
[BugLab_Argument_Swapping]^return  ( thisEnd >= thisStart && millisInstant < millisInstant ) ;^100^^^^^97^101^return  ( millisInstant >= thisStart && millisInstant < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   long millisInstant [VARIABLES] boolean  long  millisInstant  thisEnd  thisStart  
[BugLab_Argument_Swapping]^return  ( millisInstant >= thisEnd && millisInstant < thisStart ) ;^100^^^^^97^101^return  ( millisInstant >= thisStart && millisInstant < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   long millisInstant [VARIABLES] boolean  long  millisInstant  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( millisInstant >= thisStart || millisInstant < thisEnd ) ;^100^^^^^97^101^return  ( millisInstant >= thisStart && millisInstant < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   long millisInstant [VARIABLES] boolean  long  millisInstant  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( millisInstant > thisStart && millisInstant < thisEnd ) ;^100^^^^^97^101^return  ( millisInstant >= thisStart && millisInstant < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   long millisInstant [VARIABLES] boolean  long  millisInstant  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( millisInstant >= thisStart && millisInstant > thisEnd ) ;^100^^^^^97^101^return  ( millisInstant >= thisStart && millisInstant < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   long millisInstant [VARIABLES] boolean  long  millisInstant  thisEnd  thisStart  
[BugLab_Wrong_Operator]^if  ( instant != null )  {^138^^^^^137^142^if  ( instant == null )  {^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   ReadableInstant instant [VARIABLES] ReadableInstant  instant  boolean  
[BugLab_Wrong_Operator]^if  ( interval != null )  {^179^^^^^178^187^if  ( interval == null )  {^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Variable_Misuse]^return  ( thisEnd <= otherStart && otherStart < thisEnd && otherEnd <= thisEnd ) ;^186^^^^^178^187^return  ( thisStart <= otherStart && otherStart < thisEnd && otherEnd <= thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Variable_Misuse]^return  ( thisStart <= thisStart && otherStart < thisEnd && otherEnd <= thisEnd ) ;^186^^^^^178^187^return  ( thisStart <= otherStart && otherStart < thisEnd && otherEnd <= thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Variable_Misuse]^return  ( thisStart <= otherStart && otherStart < thisStart && otherEnd <= thisEnd ) ;^186^^^^^178^187^return  ( thisStart <= otherStart && otherStart < thisEnd && otherEnd <= thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Variable_Misuse]^return  ( thisStart <= otherStart && otherStart < thisEnd && thisStart <= thisEnd ) ;^186^^^^^178^187^return  ( thisStart <= otherStart && otherStart < thisEnd && otherEnd <= thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Argument_Swapping]^return  ( thisEnd <= otherStart && otherStart < thisStart && otherEnd <= thisEnd ) ;^186^^^^^178^187^return  ( thisStart <= otherStart && otherStart < thisEnd && otherEnd <= thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Argument_Swapping]^return  ( thisStart <= otherEnd && otherStart < thisEnd && otherStart <= thisEnd ) ;^186^^^^^178^187^return  ( thisStart <= otherStart && otherStart < thisEnd && otherEnd <= thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Argument_Swapping]^return  ( thisStart <= otherStart && otherStart < otherEnd && thisEnd <= thisEnd ) ;^186^^^^^178^187^return  ( thisStart <= otherStart && otherStart < thisEnd && otherEnd <= thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( thisStart <= otherStart || otherStart < thisEnd && otherEnd <= thisEnd ) ;^186^^^^^178^187^return  ( thisStart <= otherStart && otherStart < thisEnd && otherEnd <= thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( thisStart == otherStart && otherStart < thisEnd && otherEnd <= thisEnd ) ;^186^^^^^178^187^return  ( thisStart <= otherStart && otherStart < thisEnd && otherEnd <= thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( thisStart <= otherStart && otherStart == thisEnd && otherEnd <= thisEnd ) ;^186^^^^^178^187^return  ( thisStart <= otherStart && otherStart < thisEnd && otherEnd <= thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( thisStart <= otherStart && otherStart < thisEnd && otherEnd < thisEnd ) ;^186^^^^^178^187^return  ( thisStart <= otherStart && otherStart < thisEnd && otherEnd <= thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] contains [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Wrong_Operator]^if  ( interval != null )  {^233^^^^^230^241^if  ( interval == null )  {^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Variable_Misuse]^return  ( thisEnd < otherEnd && otherStart < thisEnd ) ;^239^^^^^230^241^return  ( thisStart < otherEnd && otherStart < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Variable_Misuse]^return  ( thisStart < thisEnd && otherStart < thisEnd ) ;^239^^^^^230^241^return  ( thisStart < otherEnd && otherStart < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Variable_Misuse]^return  ( thisStart < otherEnd && thisStart < thisEnd ) ;^239^^^^^230^241^return  ( thisStart < otherEnd && otherStart < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Variable_Misuse]^return  ( thisStart < otherEnd && otherStart < thisStart ) ;^239^^^^^230^241^return  ( thisStart < otherEnd && otherStart < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Argument_Swapping]^return  ( thisEnd < otherEnd && otherStart < thisStart ) ;^239^^^^^230^241^return  ( thisStart < otherEnd && otherStart < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Argument_Swapping]^return  ( thisStart < otherStart && otherEnd < thisEnd ) ;^239^^^^^230^241^return  ( thisStart < otherEnd && otherStart < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( thisStart < otherEnd || otherStart < thisEnd ) ;^239^^^^^230^241^return  ( thisStart < otherEnd && otherStart < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( thisStart <= otherEnd && otherStart < thisEnd ) ;^239^^^^^230^241^return  ( thisStart < otherEnd && otherStart < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( thisStart < otherEnd && otherStart <= thisEnd ) ;^239^^^^^230^241^return  ( thisStart < otherEnd && otherStart < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Variable_Misuse]^return  ( otherStart < now && now < thisEnd ) ;^235^^^^^230^241^return  ( thisStart < now && now < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Variable_Misuse]^return  ( thisStart < thisEnd && now < thisEnd ) ;^235^^^^^230^241^return  ( thisStart < now && now < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Variable_Misuse]^return  ( thisStart < now && now < otherStart ) ;^235^^^^^230^241^return  ( thisStart < now && now < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Argument_Swapping]^return  ( now < thisStart && now < thisEnd ) ;^235^^^^^230^241^return  ( thisStart < now && now < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Argument_Swapping]^return  ( thisStart < thisEnd && now < now ) ;^235^^^^^230^241^return  ( thisStart < now && now < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( thisStart < now || now < thisEnd ) ;^235^^^^^230^241^return  ( thisStart < now && now < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( thisStart <= now && now < thisEnd ) ;^235^^^^^230^241^return  ( thisStart < now && now < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( thisStart < now && now > thisEnd ) ;^235^^^^^230^241^return  ( thisStart < now && now < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Variable_Misuse]^return  ( thisStart < thisStart && now < thisEnd ) ;^235^^^^^230^241^return  ( thisStart < now && now < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( thisStart < now && now <= thisEnd ) ;^235^^^^^230^241^return  ( thisStart < now && now < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Argument_Swapping]^return  ( otherStart < otherEnd && thisStart < thisEnd ) ;^239^^^^^230^241^return  ( thisStart < otherEnd && otherStart < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Argument_Swapping]^return  ( thisStart < otherEnd && thisEnd < otherStart ) ;^239^^^^^230^241^return  ( thisStart < otherEnd && otherStart < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( thisStart < otherEnd && otherStart > thisEnd ) ;^239^^^^^230^241^return  ( thisStart < otherEnd && otherStart < thisEnd ) ;^[CLASS] AbstractInterval  [METHOD] overlaps [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  now  otherEnd  otherStart  thisEnd  thisStart  
[BugLab_Wrong_Operator]^return  ( getEndMillis (  )  > millisInstant ) ;^254^^^^^253^255^return  ( getEndMillis (  )  <= millisInstant ) ;^[CLASS] AbstractInterval  [METHOD] isBefore [RETURN_TYPE] boolean   long millisInstant [VARIABLES] boolean  long  millisInstant  
[BugLab_Wrong_Operator]^if  ( instant != null )  {^277^^^^^276^281^if  ( instant == null )  {^[CLASS] AbstractInterval  [METHOD] isBefore [RETURN_TYPE] boolean   ReadableInstant instant [VARIABLES] ReadableInstant  instant  boolean  
[BugLab_Wrong_Operator]^if  ( interval != null )  {^292^^^^^291^296^if  ( interval == null )  {^[CLASS] AbstractInterval  [METHOD] isBefore [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  
[BugLab_Wrong_Operator]^return  ( getStartMillis (  )  >= millisInstant ) ;^309^^^^^308^310^return  ( getStartMillis (  )  > millisInstant ) ;^[CLASS] AbstractInterval  [METHOD] isAfter [RETURN_TYPE] boolean   long millisInstant [VARIABLES] boolean  long  millisInstant  
[BugLab_Wrong_Operator]^if  ( instant != null )  {^332^^^^^331^336^if  ( instant == null )  {^[CLASS] AbstractInterval  [METHOD] isAfter [RETURN_TYPE] boolean   ReadableInstant instant [VARIABLES] ReadableInstant  instant  boolean  
[BugLab_Wrong_Operator]^if  ( interval != null )  {^349^^^^^347^355^if  ( interval == null )  {^[CLASS] AbstractInterval  [METHOD] isAfter [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  endMillis  
[BugLab_Wrong_Operator]^return  ( getStartMillis (  )  > endMillis ) ;^354^^^^^347^355^return  ( getStartMillis (  )  >= endMillis ) ;^[CLASS] AbstractInterval  [METHOD] isAfter [RETURN_TYPE] boolean   ReadableInterval interval [VARIABLES] boolean  ReadableInterval  interval  long  endMillis  
[BugLab_Wrong_Operator]^if  ( durMillis >= 0 )  {^401^^^^^399^406^if  ( durMillis == 0 )  {^[CLASS] AbstractInterval  [METHOD] toDuration [RETURN_TYPE] Duration   [VARIABLES] boolean  long  durMillis  
[BugLab_Wrong_Literal]^if  ( durMillis == 1 )  {^401^^^^^399^406^if  ( durMillis == 0 )  {^[CLASS] AbstractInterval  [METHOD] toDuration [RETURN_TYPE] Duration   [VARIABLES] boolean  long  durMillis  
[BugLab_Wrong_Operator]^if  ( this != readableInterval )  {^449^^^^^448^460^if  ( this == readableInterval )  {^[CLASS] AbstractInterval  [METHOD] equals [RETURN_TYPE] boolean   Object readableInterval [VARIABLES] boolean  ReadableInterval  other  Object  readableInterval  
[BugLab_Wrong_Literal]^return false;^450^^^^^448^460^return true;^[CLASS] AbstractInterval  [METHOD] equals [RETURN_TYPE] boolean   Object readableInterval [VARIABLES] boolean  ReadableInterval  other  Object  readableInterval  
[BugLab_Wrong_Operator]^if  ( readableInterval instanceof ReadableInterval > false )  {^452^^^^^448^460^if  ( readableInterval instanceof ReadableInterval == false )  {^[CLASS] AbstractInterval  [METHOD] equals [RETURN_TYPE] boolean   Object readableInterval [VARIABLES] boolean  ReadableInterval  other  Object  readableInterval  
[BugLab_Wrong_Operator]^if  ( readableInterval  ==  ReadableInterval == false )  {^452^^^^^448^460^if  ( readableInterval instanceof ReadableInterval == false )  {^[CLASS] AbstractInterval  [METHOD] equals [RETURN_TYPE] boolean   Object readableInterval [VARIABLES] boolean  ReadableInterval  other  Object  readableInterval  
[BugLab_Wrong_Literal]^if  ( readableInterval instanceof ReadableInterval == true )  {^452^^^^^448^460^if  ( readableInterval instanceof ReadableInterval == false )  {^[CLASS] AbstractInterval  [METHOD] equals [RETURN_TYPE] boolean   Object readableInterval [VARIABLES] boolean  ReadableInterval  other  Object  readableInterval  
[BugLab_Wrong_Literal]^return true;^453^^^^^448^460^return false;^[CLASS] AbstractInterval  [METHOD] equals [RETURN_TYPE] boolean   Object readableInterval [VARIABLES] boolean  ReadableInterval  other  Object  readableInterval  
[BugLab_Wrong_Operator]^return getStartMillis (  )  == other.getStartMillis (  )  || getEndMillis (  )  == other.getEndMillis (  )  && FieldUtils.equals ( getChronology (  ) , other.getChronology (  )  ) ;^456^457^458^459^^448^460^return getStartMillis (  )  == other.getStartMillis (  )  && getEndMillis (  )  == other.getEndMillis (  )  && FieldUtils.equals ( getChronology (  ) , other.getChronology (  )  ) ;^[CLASS] AbstractInterval  [METHOD] equals [RETURN_TYPE] boolean   Object readableInterval [VARIABLES] boolean  ReadableInterval  other  Object  readableInterval  
[BugLab_Wrong_Operator]^return getStartMillis (  )  != other.getStartMillis (  )  && getEndMillis (  )  == other.getEndMillis (  )  && FieldUtils.equals ( getChronology (  ) , other.getChronology (  )  ) ;^456^457^458^459^^448^460^return getStartMillis (  )  == other.getStartMillis (  )  && getEndMillis (  )  == other.getEndMillis (  )  && FieldUtils.equals ( getChronology (  ) , other.getChronology (  )  ) ;^[CLASS] AbstractInterval  [METHOD] equals [RETURN_TYPE] boolean   Object readableInterval [VARIABLES] boolean  ReadableInterval  other  Object  readableInterval  
[BugLab_Wrong_Operator]^return getStartMillis (  )  == other.getStartMillis (  )  && getEndMillis (  )  < other.getEndMillis (  )  && FieldUtils.equals ( getChronology (  ) , other.getChronology (  )  ) ;^456^457^458^459^^448^460^return getStartMillis (  )  == other.getStartMillis (  )  && getEndMillis (  )  == other.getEndMillis (  )  && FieldUtils.equals ( getChronology (  ) , other.getChronology (  )  ) ;^[CLASS] AbstractInterval  [METHOD] equals [RETURN_TYPE] boolean   Object readableInterval [VARIABLES] boolean  ReadableInterval  other  Object  readableInterval  
[BugLab_Wrong_Literal]^int result = result;^470^^^^^467^475^int result = 97;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Variable_Misuse]^result = 31 * result +  (  ( int )   ( end ^  ( start >>> 32 )  )  ) ;^471^^^^^467^475^result = 31 * result +  (  ( int )   ( start ^  ( start >>> 32 )  )  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Argument_Swapping]^result = 31 * start +  (  ( int )   ( result ^  ( start >>> 32 )  )  ) ;^471^^^^^467^475^result = 31 * result +  (  ( int )   ( start ^  ( start >>> 32 )  )  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Wrong_Operator]^result = 31 * result +  |  (  ( int )   ( start ^  ( start >>> 32 )  )  ) ;^471^^^^^467^475^result = 31 * result +  (  ( int )   ( start ^  ( start >>> 32 )  )  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Wrong_Operator]^result = 31 + result +  (  ( int )   ( start ^  ( start >>> 32 )  )  ) ;^471^^^^^467^475^result = 31 * result +  (  ( int )   ( start ^  ( start >>> 32 )  )  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Wrong_Operator]^result = 31 * result +  (  ( int )   ( start ^  ( start  ^  32 )  )  ) ;^471^^^^^467^475^result = 31 * result +  (  ( int )   ( start ^  ( start >>> 32 )  )  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Wrong_Literal]^result = result * result +  (  ( int )   ( start ^  ( start >>> 32 )  )  ) ;^471^^^^^467^475^result = 31 * result +  (  ( int )   ( start ^  ( start >>> 32 )  )  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Variable_Misuse]^result = 31 * result +  (  ( int )   ( start ^  ( end >>> 32 )  )  ) ;^472^^^^^467^475^result = 31 * result +  (  ( int )   ( end ^  ( end >>> 32 )  )  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Argument_Swapping]^result = 31 * end +  (  ( int )   ( result ^  ( end >>> 32 )  )  ) ;^472^^^^^467^475^result = 31 * result +  (  ( int )   ( end ^  ( end >>> 32 )  )  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Wrong_Operator]^result = 31 * result +  |  (  ( int )   ( end ^  ( end >>> 32 )  )  ) ;^472^^^^^467^475^result = 31 * result +  (  ( int )   ( end ^  ( end >>> 32 )  )  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Wrong_Operator]^result = 31 + result +  (  ( int )   ( end ^  ( end >>> 32 )  )  ) ;^472^^^^^467^475^result = 31 * result +  (  ( int )   ( end ^  ( end >>> 32 )  )  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Wrong_Operator]^result = 31 * result +  (  ( int )   ( end ^  ( end  <  32 )  )  ) ;^472^^^^^467^475^result = 31 * result +  (  ( int )   ( end ^  ( end >>> 32 )  )  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Wrong_Literal]^result = result * result +  (  ( int )   ( end ^  ( end >>> 32 )  )  ) ;^472^^^^^467^475^result = 31 * result +  (  ( int )   ( end ^  ( end >>> 32 )  )  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Wrong_Literal]^result = 31 * result +  (  ( int )   ( end ^  ( end >>> result )  )  ) ;^472^^^^^467^475^result = 31 * result +  (  ( int )   ( end ^  ( end >>> 32 )  )  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Wrong_Operator]^result = 31 * result + getChronology (  ^  ) .hashCode (  ) ;^473^^^^^467^475^result = 31 * result + getChronology (  ) .hashCode (  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Wrong_Operator]^result = 31 / result + getChronology (  ) .hashCode (  ) ;^473^^^^^467^475^result = 31 * result + getChronology (  ) .hashCode (  ) ;^[CLASS] AbstractInterval  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  end  start  int  result  
[BugLab_Wrong_Literal]^StringBuffer buf = new StringBuffer ( 47 ) ;^485^^^^^482^490^StringBuffer buf = new StringBuffer ( 48 ) ;^[CLASS] AbstractInterval  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  DateTimeFormatter  printer  