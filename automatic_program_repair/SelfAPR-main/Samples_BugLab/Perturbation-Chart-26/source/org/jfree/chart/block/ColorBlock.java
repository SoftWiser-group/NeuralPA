[BugLab_Wrong_Operator]^if  ( paint != null )  {^75^^^^^74^81^if  ( paint == null )  {^[CLASS] ColorBlock  [METHOD] <init> [RETURN_TYPE] Paint,double,double)   Paint paint double width double height [VARIABLES] double  height  width  Paint  paint  boolean  
[BugLab_Variable_Misuse]^setWidth ( height ) ;^79^^^^^74^81^setWidth ( width ) ;^[CLASS] ColorBlock  [METHOD] <init> [RETURN_TYPE] Paint,double,double)   Paint paint double width double height [VARIABLES] double  height  width  Paint  paint  boolean  
[BugLab_Variable_Misuse]^setHeight ( width ) ;^80^^^^^74^81^setHeight ( height ) ;^[CLASS] ColorBlock  [METHOD] <init> [RETURN_TYPE] Paint,double,double)   Paint paint double width double height [VARIABLES] double  height  width  Paint  paint  boolean  
[BugLab_Variable_Misuse]^return paint;^91^^^^^90^92^return this.paint;^[CLASS] ColorBlock  [METHOD] getPaint [RETURN_TYPE] Paint   [VARIABLES] Paint  paint  boolean  
[BugLab_Variable_Misuse]^g2.setPaint ( paint ) ;^102^^^^^100^104^g2.setPaint ( this.paint ) ;^[CLASS] ColorBlock  [METHOD] draw [RETURN_TYPE] void   Graphics2D g2 Rectangle2D area [VARIABLES] Rectangle2D  area  bounds  Paint  paint  boolean  Graphics2D  g2  
[BugLab_Variable_Misuse]^g2.fill ( area ) ;^103^^^^^100^104^g2.fill ( bounds ) ;^[CLASS] ColorBlock  [METHOD] draw [RETURN_TYPE] void   Graphics2D g2 Rectangle2D area [VARIABLES] Rectangle2D  area  bounds  Paint  paint  boolean  Graphics2D  g2  
[BugLab_Argument_Swapping]^draw ( area, g2 ) ;^116^^^^^115^118^draw ( g2, area ) ;^[CLASS] ColorBlock  [METHOD] draw [RETURN_TYPE] Object   Graphics2D g2 Rectangle2D area Object params [VARIABLES] Rectangle2D  area  Object  params  Paint  paint  boolean  Graphics2D  g2  
[BugLab_Wrong_Operator]^if  ( obj != this )  {^128^^^^^127^139^if  ( obj == this )  {^[CLASS] ColorBlock  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Paint  paint  boolean  ColorBlock  that  
[BugLab_Wrong_Literal]^return false;^129^^^^^127^139^return true;^[CLASS] ColorBlock  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Paint  paint  boolean  ColorBlock  that  
[BugLab_Wrong_Operator]^if  ( ! ( obj  >=  ColorBlock )  )  {^131^^^^^127^139^if  ( ! ( obj instanceof ColorBlock )  )  {^[CLASS] ColorBlock  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Paint  paint  boolean  ColorBlock  that  
[BugLab_Wrong_Literal]^return true;^132^^^^^127^139^return false;^[CLASS] ColorBlock  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Paint  paint  boolean  ColorBlock  that  
[BugLab_Variable_Misuse]^if  ( !PaintUtilities.equal ( paint, that.paint )  )  {^135^^^^^127^139^if  ( !PaintUtilities.equal ( this.paint, that.paint )  )  {^[CLASS] ColorBlock  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Paint  paint  boolean  ColorBlock  that  
[BugLab_Variable_Misuse]^if  ( !PaintUtilities.equal ( this.paint, paint )  )  {^135^^^^^127^139^if  ( !PaintUtilities.equal ( this.paint, that.paint )  )  {^[CLASS] ColorBlock  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Paint  paint  boolean  ColorBlock  that  
[BugLab_Argument_Swapping]^if  ( !PaintUtilities.equal ( this.paint, that.paint.paint )  )  {^135^^^^^127^139^if  ( !PaintUtilities.equal ( this.paint, that.paint )  )  {^[CLASS] ColorBlock  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Paint  paint  boolean  ColorBlock  that  
[BugLab_Argument_Swapping]^if  ( !PaintUtilities.equal ( that, this.paint.paint )  )  {^135^^^^^127^139^if  ( !PaintUtilities.equal ( this.paint, that.paint )  )  {^[CLASS] ColorBlock  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Paint  paint  boolean  ColorBlock  that  
[BugLab_Argument_Swapping]^if  ( !PaintUtilities.equal ( that.paint, this.paint )  )  {^135^^^^^127^139^if  ( !PaintUtilities.equal ( this.paint, that.paint )  )  {^[CLASS] ColorBlock  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Paint  paint  boolean  ColorBlock  that  
[BugLab_Wrong_Literal]^return true;^136^^^^^127^139^return false;^[CLASS] ColorBlock  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Paint  paint  boolean  ColorBlock  that  
[BugLab_Variable_Misuse]^SerialUtilities.writePaint ( paint, stream ) ;^150^^^^^148^151^SerialUtilities.writePaint ( this.paint, stream ) ;^[CLASS] ColorBlock  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream stream [VARIABLES] ObjectOutputStream  stream  Paint  paint  boolean  
[BugLab_Argument_Swapping]^SerialUtilities.writePaint ( stream, this.paint ) ;^150^^^^^148^151^SerialUtilities.writePaint ( this.paint, stream ) ;^[CLASS] ColorBlock  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream stream [VARIABLES] ObjectOutputStream  stream  Paint  paint  boolean  
