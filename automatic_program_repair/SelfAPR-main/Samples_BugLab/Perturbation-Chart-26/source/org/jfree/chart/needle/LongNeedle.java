[BugLab_Variable_Misuse]^float midX =  ( float )   ( y +  ( plotArea.getWidth (  )  * 0.5 )  ) ;^93^^^^^78^108^float midX =  ( float )   ( minX +  ( plotArea.getWidth (  )  * 0.5 )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^float midX =  ( float )   ( plotArea +  ( minX.getWidth (  )  * 0.5 )  ) ;^93^^^^^78^108^float midX =  ( float )   ( minX +  ( plotArea.getWidth (  )  * 0.5 )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^float midX =  <<  ( float )   ( minX +  ( plotArea.getWidth (  )  * 0.5 )  ) ;^93^^^^^78^108^float midX =  ( float )   ( minX +  ( plotArea.getWidth (  )  * 0.5 )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^float midX =  ( float )   ( minX +  ( plotArea.getWidth (  )  - 0.5 )  ) ;^93^^^^^78^108^float midX =  ( float )   ( minX +  ( plotArea.getWidth (  )  * 0.5 )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^float midY =  ( float )   ( y +  ( plotArea.getHeight (  )  * 0.8 )  ) ;^94^^^^^79^109^float midY =  ( float )   ( minY +  ( plotArea.getHeight (  )  * 0.8 )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^float midY =  ( float )   ( plotArea +  ( minY.getHeight (  )  * 0.8 )  ) ;^94^^^^^79^109^float midY =  ( float )   ( minY +  ( plotArea.getHeight (  )  * 0.8 )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^float midY =  >  ( float )   ( minY +  ( plotArea.getHeight (  )  * 0.8 )  ) ;^94^^^^^79^109^float midY =  ( float )   ( minY +  ( plotArea.getHeight (  )  * 0.8 )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^float midY =  ( float )   ( minY +  ( plotArea.getHeight (  )  / 0.8 )  ) ;^94^^^^^79^109^float midY =  ( float )   ( minY +  ( plotArea.getHeight (  )  * 0.8 )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^float y = minY -  ( 2 *  ( maxY - midY )  ) ;^95^^^^^80^110^float y = maxY -  ( 2 *  ( maxY - midY )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^float y = maxY -  ( 2 *  ( maxY - minY )  ) ;^95^^^^^80^110^float y = maxY -  ( 2 *  ( maxY - midY )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^float y = midY -  ( 2 *  ( maxY - maxY )  ) ;^95^^^^^80^110^float y = maxY -  ( 2 *  ( maxY - midY )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^float y = maxY  >=   ( 2 *  ( maxY - midY )  ) ;^95^^^^^80^110^float y = maxY -  ( 2 *  ( maxY - midY )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^float / y = maxY -  ( 2 *  ( maxY - midY )  ) ;^95^^^^^80^110^float y = maxY -  ( 2 *  ( maxY - midY )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^float y = maxY -  ( 2 *  ( maxY  ||  midY )  ) ;^95^^^^^80^110^float y = maxY -  ( 2 *  ( maxY - midY )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Literal]^float y = maxY -  ( 3 *  ( maxY - midY )  ) ;^95^^^^^80^110^float y = maxY -  ( 2 *  ( maxY - midY )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^if  ( minX < minY )  {^96^^^^^81^111^if  ( y < minY )  {^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^if  ( y < minX )  {^96^^^^^81^111^if  ( y < minY )  {^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^if  ( minY < y )  {^96^^^^^81^111^if  ( y < minY )  {^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^if  ( y <= minY )  {^96^^^^^81^111^if  ( y < minY )  {^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^y = midY;^97^^^^^82^112^y = minY;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^y = minX;^97^^^^^82^112^y = minY;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape1.moveTo ( y, midY ) ;^99^^^^^84^114^shape1.moveTo ( minX, midY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape1.moveTo ( minX, y ) ;^99^^^^^84^114^shape1.moveTo ( minX, midY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^shape1.moveTo ( midY, minX ) ;^99^^^^^84^114^shape1.moveTo ( minX, midY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape1.lineTo ( minX, minY ) ;^100^^^^^85^115^shape1.lineTo ( midX, minY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape1.lineTo ( midX, y ) ;^100^^^^^85^115^shape1.lineTo ( midX, minY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^shape1.lineTo ( minY, midX ) ;^100^^^^^85^115^shape1.lineTo ( midX, minY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape1.lineTo ( midX, midY ) ;^101^^^^^86^116^shape1.lineTo ( midX, y ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^shape1.lineTo ( y, midX ) ;^101^^^^^86^116^shape1.lineTo ( midX, y ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape2.moveTo ( minY, midY ) ;^104^^^^^89^119^shape2.moveTo ( maxX, midY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape2.moveTo ( maxX, y ) ;^104^^^^^89^119^shape2.moveTo ( maxX, midY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^shape2.moveTo ( midY, maxX ) ;^104^^^^^89^119^shape2.moveTo ( maxX, midY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape2.lineTo ( minX, minY ) ;^105^^^^^90^120^shape2.lineTo ( midX, minY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape2.lineTo ( midX, y ) ;^105^^^^^90^120^shape2.lineTo ( midX, minY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^shape2.lineTo ( minY, midX ) ;^105^^^^^90^120^shape2.lineTo ( midX, minY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape2.lineTo ( minY, y ) ;^106^^^^^91^121^shape2.lineTo ( midX, y ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape2.lineTo ( midX, minY ) ;^106^^^^^91^121^shape2.lineTo ( midX, y ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^shape2.lineTo ( y, midX ) ;^106^^^^^91^121^shape2.lineTo ( midX, y ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape3.moveTo ( minY, midY ) ;^109^^^^^94^124^shape3.moveTo ( minX, midY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape3.moveTo ( minX, y ) ;^109^^^^^94^124^shape3.moveTo ( minX, midY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^shape3.moveTo ( midY, minX ) ;^109^^^^^94^124^shape3.moveTo ( minX, midY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape3.lineTo ( minY, maxY ) ;^110^^^^^95^125^shape3.lineTo ( midX, maxY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape3.lineTo ( midX, minY ) ;^110^^^^^95^125^shape3.lineTo ( midX, maxY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^shape3.lineTo ( maxY, midX ) ;^110^^^^^95^125^shape3.lineTo ( midX, maxY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape3.lineTo ( y, midY ) ;^111^^^^^96^126^shape3.lineTo ( maxX, midY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape3.lineTo ( maxX, y ) ;^111^^^^^96^126^shape3.lineTo ( maxX, midY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^shape3.lineTo ( midY, maxX ) ;^111^^^^^96^126^shape3.lineTo ( maxX, midY ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^shape3.lineTo ( minY, y ) ;^112^^^^^97^127^shape3.lineTo ( midX, y ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^shape3.lineTo ( y, midX ) ;^112^^^^^97^127^shape3.lineTo ( midX, y ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^Shape s1 = shape3;^115^^^^^100^130^Shape s1 = shape1;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^Shape s2 = shape3;^116^^^^^101^131^Shape s2 = shape2;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^Shape s3 = shape2;^117^^^^^102^132^Shape s3 = shape3;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^if  (  ( rotate != null )  ||  ( angle != 0 )  )  {^119^^^^^104^134^if  (  ( rotate != null )  &&  ( angle != 0 )  )  {^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^if  (  ( rotate == null )  &&  ( angle != 0 )  )  {^119^^^^^104^134^if  (  ( rotate != null )  &&  ( angle != 0 )  )  {^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^if  (  ( rotate != null )  &&  ( angle == 0 )  )  {^119^^^^^104^134^if  (  ( rotate != null )  &&  ( angle != 0 )  )  {^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Literal]^if  (  ( rotate != null )  &&  ( angle != -1 )  )  {^119^^^^^104^134^if  (  ( rotate != null )  &&  ( angle != 0 )  )  {^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^s1 = shape3.createTransformedShape ( transform ) ;^122^^^^^107^137^s1 = shape1.createTransformedShape ( transform ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^s1 = transform.createTransformedShape ( shape1 ) ;^122^^^^^107^137^s1 = shape1.createTransformedShape ( transform ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^s2 = transform.createTransformedShape ( shape2 ) ;^123^^^^^108^138^s2 = shape2.createTransformedShape ( transform ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^s3 = shape2.createTransformedShape ( transform ) ;^124^^^^^109^139^s3 = shape3.createTransformedShape ( transform ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^s3 = transform.createTransformedShape ( shape3 ) ;^124^^^^^109^139^s3 = shape3.createTransformedShape ( transform ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Argument_Swapping]^getTransform (  ) .setToRotation ( rotate, angle.getX (  ) , rotate.getY (  )  ) ;^121^^^^^106^136^getTransform (  ) .setToRotation ( angle, rotate.getX (  ) , rotate.getY (  )  ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^s2 = shape3.createTransformedShape ( transform ) ;^123^^^^^108^138^s2 = shape2.createTransformedShape ( transform ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^s2 = shape2.createTransformedShape ( null ) ;^123^^^^^108^138^s2 = shape2.createTransformedShape ( transform ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^if  ( getHighlightPaint (  )  == null )  {^128^^^^^113^143^if  ( getHighlightPaint (  )  != null )  {^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^g2.fill ( s2 ) ;^130^^^^^115^145^g2.fill ( s3 ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^if  ( getFillPaint (  )  == null )  {^133^^^^^118^148^if  ( getFillPaint (  )  != null )  {^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^g2.fill ( s3 ) ;^135^^^^^120^150^g2.fill ( s1 ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^g2.fill ( s3 ) ;^136^^^^^121^151^g2.fill ( s2 ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^if  ( getOutlinePaint (  )  == null )  {^140^^^^^125^155^if  ( getOutlinePaint (  )  != null )  {^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^g2.draw ( s3 ) ;^143^^^^^128^158^g2.draw ( s1 ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^g2.draw ( s3 ) ;^144^^^^^129^159^g2.draw ( s2 ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Variable_Misuse]^g2.draw ( s2 ) ;^145^^^^^130^160^g2.draw ( s3 ) ;^[CLASS] LongNeedle  [METHOD] drawNeedle [RETURN_TYPE] void   Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [VARIABLES] Shape  s1  s2  s3  boolean  GeneralPath  shape1  shape2  shape3  Point2D  rotate  double  angle  Rectangle2D  plotArea  float  maxX  maxY  midX  midY  minX  minY  y  long  serialVersionUID  Graphics2D  g2  
[BugLab_Wrong_Operator]^if  ( obj != this )  {^157^^^^^156^167^if  ( obj == this )  {^[CLASS] LongNeedle  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] long  serialVersionUID  Object  obj  boolean  
[BugLab_Wrong_Literal]^return false;^158^^^^^156^167^return true;^[CLASS] LongNeedle  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] long  serialVersionUID  Object  obj  boolean  
[BugLab_Wrong_Operator]^if  ( ! ( obj  <<  LongNeedle )  )  {^160^^^^^156^167^if  ( ! ( obj instanceof LongNeedle )  )  {^[CLASS] LongNeedle  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] long  serialVersionUID  Object  obj  boolean  
[BugLab_Wrong_Literal]^return true;^161^^^^^156^167^return false;^[CLASS] LongNeedle  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] long  serialVersionUID  Object  obj  boolean  
[BugLab_Wrong_Literal]^return true;^164^^^^^156^167^return false;^[CLASS] LongNeedle  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] long  serialVersionUID  Object  obj  boolean  
[BugLab_Wrong_Literal]^return false;^166^^^^^156^167^return true;^[CLASS] LongNeedle  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] long  serialVersionUID  Object  obj  boolean  