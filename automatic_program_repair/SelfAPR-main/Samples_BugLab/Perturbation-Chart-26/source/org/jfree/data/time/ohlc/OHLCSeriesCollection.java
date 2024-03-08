[BugLab_Wrong_Operator]^if  ( series != null )  {^85^^^^^84^91^if  ( series == null )  {^[CLASS] OHLCSeriesCollection  [METHOD] addSeries [RETURN_TYPE] void   OHLCSeries series [VARIABLES] List  data  TimePeriodAnchor  xPosition  boolean  OHLCSeries  series  
[BugLab_Variable_Misuse]^return data.size (  ) ;^99^^^^^98^100^return this.data.size (  ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getSeriesCount [RETURN_TYPE] int   [VARIABLES] List  data  TimePeriodAnchor  xPosition  boolean  
[BugLab_Wrong_Operator]^if  (  ( series < 0 )  &&  ( series >= getSeriesCount (  )  )  )  {^113^^^^^112^117^if  (  ( series < 0 )  ||  ( series >= getSeriesCount (  )  )  )  {^[CLASS] OHLCSeriesCollection  [METHOD] getSeries [RETURN_TYPE] OHLCSeries   int series [VARIABLES] List  data  TimePeriodAnchor  xPosition  boolean  int  series  
[BugLab_Wrong_Operator]^if  (  ( series == 0 )  ||  ( series >= getSeriesCount (  )  )  )  {^113^^^^^112^117^if  (  ( series < 0 )  ||  ( series >= getSeriesCount (  )  )  )  {^[CLASS] OHLCSeriesCollection  [METHOD] getSeries [RETURN_TYPE] OHLCSeries   int series [VARIABLES] List  data  TimePeriodAnchor  xPosition  boolean  int  series  
[BugLab_Wrong_Operator]^if  (  ( series < 0 )  ||  ( series < getSeriesCount (  )  )  )  {^113^^^^^112^117^if  (  ( series < 0 )  ||  ( series >= getSeriesCount (  )  )  )  {^[CLASS] OHLCSeriesCollection  [METHOD] getSeries [RETURN_TYPE] OHLCSeries   int series [VARIABLES] List  data  TimePeriodAnchor  xPosition  boolean  int  series  
[BugLab_Wrong_Literal]^if  (  ( series < series )  ||  ( series >= getSeriesCount (  )  )  )  {^113^^^^^112^117^if  (  ( series < 0 )  ||  ( series >= getSeriesCount (  )  )  )  {^[CLASS] OHLCSeriesCollection  [METHOD] getSeries [RETURN_TYPE] OHLCSeries   int series [VARIABLES] List  data  TimePeriodAnchor  xPosition  boolean  int  series  
[BugLab_Variable_Misuse]^return  ( OHLCSeries )  data.get ( series ) ;^116^^^^^112^117^return  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getSeries [RETURN_TYPE] OHLCSeries   int series [VARIABLES] List  data  TimePeriodAnchor  xPosition  boolean  int  series  
[BugLab_Argument_Swapping]^return  ( OHLCSeries )  series.get ( this.data ) ;^116^^^^^112^117^return  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getSeries [RETURN_TYPE] OHLCSeries   int series [VARIABLES] List  data  TimePeriodAnchor  xPosition  boolean  int  series  
[BugLab_Variable_Misuse]^if  ( xPosition == TimePeriodAnchor.START )  {^159^^^^^157^169^if  ( this.xPosition == TimePeriodAnchor.START )  {^[CLASS] OHLCSeriesCollection  [METHOD] getX [RETURN_TYPE] long   RegularTimePeriod period [VARIABLES] List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  boolean  long  result  
[BugLab_Wrong_Operator]^if  ( this.xPosition > TimePeriodAnchor.START )  {^159^^^^^157^169^if  ( this.xPosition == TimePeriodAnchor.START )  {^[CLASS] OHLCSeriesCollection  [METHOD] getX [RETURN_TYPE] long   RegularTimePeriod period [VARIABLES] List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  boolean  long  result  
[BugLab_Wrong_Operator]^if  ( this.xPosition <= TimePeriodAnchor.START )  {^159^^^^^157^169^if  ( this.xPosition == TimePeriodAnchor.START )  {^[CLASS] OHLCSeriesCollection  [METHOD] getX [RETURN_TYPE] long   RegularTimePeriod period [VARIABLES] List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  boolean  long  result  
[BugLab_Variable_Misuse]^else if  ( xPosition == TimePeriodAnchor.MIDDLE )  {^162^^^^^157^169^else if  ( this.xPosition == TimePeriodAnchor.MIDDLE )  {^[CLASS] OHLCSeriesCollection  [METHOD] getX [RETURN_TYPE] long   RegularTimePeriod period [VARIABLES] List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  boolean  long  result  
[BugLab_Wrong_Operator]^else if  ( this.xPosition <= TimePeriodAnchor.MIDDLE )  {^162^^^^^157^169^else if  ( this.xPosition == TimePeriodAnchor.MIDDLE )  {^[CLASS] OHLCSeriesCollection  [METHOD] getX [RETURN_TYPE] long   RegularTimePeriod period [VARIABLES] List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  boolean  long  result  
[BugLab_Variable_Misuse]^else if  ( xPosition == TimePeriodAnchor.END )  {^165^^^^^157^169^else if  ( this.xPosition == TimePeriodAnchor.END )  {^[CLASS] OHLCSeriesCollection  [METHOD] getX [RETURN_TYPE] long   RegularTimePeriod period [VARIABLES] List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  boolean  long  result  
[BugLab_Wrong_Operator]^else if  ( this.xPosition != TimePeriodAnchor.END )  {^165^^^^^157^169^else if  ( this.xPosition == TimePeriodAnchor.END )  {^[CLASS] OHLCSeriesCollection  [METHOD] getX [RETURN_TYPE] long   RegularTimePeriod period [VARIABLES] List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  boolean  long  result  
[BugLab_Wrong_Operator]^else if  ( this.xPosition <= TimePeriodAnchor.END )  {^165^^^^^157^169^else if  ( this.xPosition == TimePeriodAnchor.END )  {^[CLASS] OHLCSeriesCollection  [METHOD] getX [RETURN_TYPE] long   RegularTimePeriod period [VARIABLES] List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  boolean  long  result  
[BugLab_Wrong_Operator]^else if  ( this.xPosition >= TimePeriodAnchor.MIDDLE )  {^162^^^^^157^169^else if  ( this.xPosition == TimePeriodAnchor.MIDDLE )  {^[CLASS] OHLCSeriesCollection  [METHOD] getX [RETURN_TYPE] long   RegularTimePeriod period [VARIABLES] List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  boolean  long  result  
[BugLab_Wrong_Operator]^else if  ( this.xPosition != TimePeriodAnchor.MIDDLE )  {^162^^^^^157^169^else if  ( this.xPosition == TimePeriodAnchor.MIDDLE )  {^[CLASS] OHLCSeriesCollection  [METHOD] getX [RETURN_TYPE] long   RegularTimePeriod period [VARIABLES] List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  boolean  long  result  
[BugLab_Variable_Misuse]^OHLCSeries s =  ( OHLCSeries )  this.data.get ( item ) ;^180^^^^^179^184^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getXValue [RETURN_TYPE] double   int series int item [VARIABLES] OHLCItem  di  boolean  List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  OHLCSeries  s  int  item  series  
[BugLab_Variable_Misuse]^OHLCSeries s =  ( OHLCSeries )  data.get ( series ) ;^180^^^^^179^184^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getXValue [RETURN_TYPE] double   int series int item [VARIABLES] OHLCItem  di  boolean  List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^OHLCSeries s =  ( OHLCSeries )  series.get ( this.data ) ;^180^^^^^179^184^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getXValue [RETURN_TYPE] double   int series int item [VARIABLES] OHLCItem  di  boolean  List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  OHLCSeries  s  int  item  series  
[BugLab_Variable_Misuse]^OHLCItem di =  ( OHLCItem )  s.getDataItem ( series ) ;^181^^^^^179^184^OHLCItem di =  ( OHLCItem )  s.getDataItem ( item ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getXValue [RETURN_TYPE] double   int series int item [VARIABLES] OHLCItem  di  boolean  List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^OHLCItem di =  ( OHLCItem )  item.getDataItem ( s ) ;^181^^^^^179^184^OHLCItem di =  ( OHLCItem )  s.getDataItem ( item ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getXValue [RETURN_TYPE] double   int series int item [VARIABLES] OHLCItem  di  boolean  List  data  TimePeriodAnchor  xPosition  RegularTimePeriod  period  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^return new Double ( getXValue ( item, series )  ) ;^195^^^^^194^196^return new Double ( getXValue ( series, item )  ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getX [RETURN_TYPE] Number   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  boolean  int  item  series  
[BugLab_Variable_Misuse]^OHLCSeries s =  ( OHLCSeries )  this.data.get ( item ) ;^207^^^^^206^210^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getY [RETURN_TYPE] Number   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^OHLCSeries s =  ( OHLCSeries )  series.get ( this.data ) ;^207^^^^^206^210^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getY [RETURN_TYPE] Number   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Variable_Misuse]^OHLCSeries s =  ( OHLCSeries )  data.get ( series ) ;^207^^^^^206^210^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getY [RETURN_TYPE] Number   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Variable_Misuse]^OHLCItem di =  ( OHLCItem )  s.getDataItem ( series ) ;^208^^^^^206^210^OHLCItem di =  ( OHLCItem )  s.getDataItem ( item ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getY [RETURN_TYPE] Number   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^OHLCItem di =  ( OHLCItem )  item.getDataItem ( s ) ;^208^^^^^206^210^OHLCItem di =  ( OHLCItem )  s.getDataItem ( item ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getY [RETURN_TYPE] Number   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Variable_Misuse]^OHLCSeries s =  ( OHLCSeries )  this.data.get ( item ) ;^221^^^^^220^224^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getOpenValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Variable_Misuse]^OHLCSeries s =  ( OHLCSeries )  data.get ( series ) ;^221^^^^^220^224^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getOpenValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^OHLCSeries s =  ( OHLCSeries )  series.get ( this.data ) ;^221^^^^^220^224^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getOpenValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^OHLCItem di =  ( OHLCItem )  item.getDataItem ( s ) ;^222^^^^^220^224^OHLCItem di =  ( OHLCItem )  s.getDataItem ( item ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getOpenValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^return new Double ( getOpenValue ( item, series )  ) ;^235^^^^^234^236^return new Double ( getOpenValue ( series, item )  ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getOpen [RETURN_TYPE] Number   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  boolean  int  item  series  
[BugLab_Variable_Misuse]^OHLCSeries s =  ( OHLCSeries )  data.get ( series ) ;^247^^^^^246^250^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getCloseValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^OHLCSeries s =  ( OHLCSeries )  series.get ( this.data ) ;^247^^^^^246^250^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getCloseValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Variable_Misuse]^OHLCSeries s =  ( OHLCSeries )  this.data.get ( item ) ;^247^^^^^246^250^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getCloseValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Variable_Misuse]^OHLCItem di =  ( OHLCItem )  s.getDataItem ( series ) ;^248^^^^^246^250^OHLCItem di =  ( OHLCItem )  s.getDataItem ( item ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getCloseValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^OHLCItem di =  ( OHLCItem )  item.getDataItem ( s ) ;^248^^^^^246^250^OHLCItem di =  ( OHLCItem )  s.getDataItem ( item ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getCloseValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^return new Double ( getCloseValue ( item, series )  ) ;^261^^^^^260^262^return new Double ( getCloseValue ( series, item )  ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getClose [RETURN_TYPE] Number   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  boolean  int  item  series  
[BugLab_Variable_Misuse]^OHLCSeries s =  ( OHLCSeries )  this.data.get ( item ) ;^273^^^^^272^276^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getHighValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Variable_Misuse]^OHLCSeries s =  ( OHLCSeries )  data.get ( series ) ;^273^^^^^272^276^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getHighValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^OHLCSeries s =  ( OHLCSeries )  series.get ( this.data ) ;^273^^^^^272^276^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getHighValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Variable_Misuse]^OHLCItem di =  ( OHLCItem )  s.getDataItem ( series ) ;^274^^^^^272^276^OHLCItem di =  ( OHLCItem )  s.getDataItem ( item ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getHighValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^OHLCItem di =  ( OHLCItem )  item.getDataItem ( s ) ;^274^^^^^272^276^OHLCItem di =  ( OHLCItem )  s.getDataItem ( item ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getHighValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^return new Double ( getHighValue ( item, series )  ) ;^287^^^^^286^288^return new Double ( getHighValue ( series, item )  ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getHigh [RETURN_TYPE] Number   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  boolean  int  item  series  
[BugLab_Variable_Misuse]^OHLCSeries s =  ( OHLCSeries )  this.data.get ( item ) ;^299^^^^^298^302^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getLowValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Variable_Misuse]^OHLCSeries s =  ( OHLCSeries )  data.get ( series ) ;^299^^^^^298^302^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getLowValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^OHLCSeries s =  ( OHLCSeries )  series.get ( this.data ) ;^299^^^^^298^302^OHLCSeries s =  ( OHLCSeries )  this.data.get ( series ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getLowValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Variable_Misuse]^OHLCItem di =  ( OHLCItem )  s.getDataItem ( series ) ;^300^^^^^298^302^OHLCItem di =  ( OHLCItem )  s.getDataItem ( item ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getLowValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^OHLCItem di =  ( OHLCItem )  item.getDataItem ( s ) ;^300^^^^^298^302^OHLCItem di =  ( OHLCItem )  s.getDataItem ( item ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getLowValue [RETURN_TYPE] double   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  OHLCItem  di  boolean  OHLCSeries  s  int  item  series  
[BugLab_Argument_Swapping]^return new Double ( getLowValue ( item, series )  ) ;^313^^^^^312^314^return new Double ( getLowValue ( series, item )  ) ;^[CLASS] OHLCSeriesCollection  [METHOD] getLow [RETURN_TYPE] Number   int series int item [VARIABLES] List  data  TimePeriodAnchor  xPosition  boolean  int  item  series  
[BugLab_Wrong_Operator]^if  ( obj > this )  {^332^^^^^331^340^if  ( obj == this )  {^[CLASS] OHLCSeriesCollection  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  data  TimePeriodAnchor  xPosition  Object  obj  boolean  OHLCSeriesCollection  that  
[BugLab_Wrong_Literal]^return false;^333^^^^^331^340^return true;^[CLASS] OHLCSeriesCollection  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  data  TimePeriodAnchor  xPosition  Object  obj  boolean  OHLCSeriesCollection  that  
[BugLab_Wrong_Operator]^if  ( ! ( obj  <<  OHLCSeriesCollection )  )  {^335^^^^^331^340^if  ( ! ( obj instanceof OHLCSeriesCollection )  )  {^[CLASS] OHLCSeriesCollection  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  data  TimePeriodAnchor  xPosition  Object  obj  boolean  OHLCSeriesCollection  that  
[BugLab_Wrong_Literal]^return true;^336^^^^^331^340^return false;^[CLASS] OHLCSeriesCollection  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  data  TimePeriodAnchor  xPosition  Object  obj  boolean  OHLCSeriesCollection  that  
[BugLab_Variable_Misuse]^return ObjectUtilities.equal ( data, that.data ) ;^339^^^^^331^340^return ObjectUtilities.equal ( this.data, that.data ) ;^[CLASS] OHLCSeriesCollection  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  data  TimePeriodAnchor  xPosition  Object  obj  boolean  OHLCSeriesCollection  that  
[BugLab_Variable_Misuse]^return ObjectUtilities.equal ( this.data, data ) ;^339^^^^^331^340^return ObjectUtilities.equal ( this.data, that.data ) ;^[CLASS] OHLCSeriesCollection  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  data  TimePeriodAnchor  xPosition  Object  obj  boolean  OHLCSeriesCollection  that  
[BugLab_Argument_Swapping]^return ObjectUtilities.equal ( that, this.data.data ) ;^339^^^^^331^340^return ObjectUtilities.equal ( this.data, that.data ) ;^[CLASS] OHLCSeriesCollection  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  data  TimePeriodAnchor  xPosition  Object  obj  boolean  OHLCSeriesCollection  that  
[BugLab_Argument_Swapping]^return ObjectUtilities.equal ( that.data, this.data ) ;^339^^^^^331^340^return ObjectUtilities.equal ( this.data, that.data ) ;^[CLASS] OHLCSeriesCollection  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  data  TimePeriodAnchor  xPosition  Object  obj  boolean  OHLCSeriesCollection  that  
[BugLab_Argument_Swapping]^return ObjectUtilities.equal ( this.data, that.data.data ) ;^339^^^^^331^340^return ObjectUtilities.equal ( this.data, that.data ) ;^[CLASS] OHLCSeriesCollection  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] List  data  TimePeriodAnchor  xPosition  Object  obj  boolean  OHLCSeriesCollection  that  
[BugLab_Variable_Misuse]^clone.data =  ( List )  ObjectUtilities.deepClone ( data ) ;^352^^^^^349^354^clone.data =  ( List )  ObjectUtilities.deepClone ( this.data ) ;^[CLASS] OHLCSeriesCollection  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] List  data  TimePeriodAnchor  xPosition  boolean  OHLCSeriesCollection  clone  