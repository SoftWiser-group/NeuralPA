[P8_Replace_Mix]^this.paint =  null;^74^^^^^73^76^this.paint = paint;^[CLASS] PaintSample  [METHOD] <init> [RETURN_TYPE] Paint)   Paint paint [VARIABLES] Paint  paint  Dimension  preferredSize  boolean  
[P3_Replace_Literal]^this.preferredSize = new Dimension ( 76, 12 ) ;^75^^^^^73^76^this.preferredSize = new Dimension ( 80, 12 ) ;^[CLASS] PaintSample  [METHOD] <init> [RETURN_TYPE] Paint)   Paint paint [VARIABLES] Paint  paint  Dimension  preferredSize  boolean  
[P3_Replace_Literal]^this.preferredSize = new Dimension ( 80, 6 ) ;^75^^^^^73^76^this.preferredSize = new Dimension ( 80, 12 ) ;^[CLASS] PaintSample  [METHOD] <init> [RETURN_TYPE] Paint)   Paint paint [VARIABLES] Paint  paint  Dimension  preferredSize  boolean  
[P8_Replace_Mix]^this.preferredSize = new Dimension ( 80L, 12 ) ;^75^^^^^73^76^this.preferredSize = new Dimension ( 80, 12 ) ;^[CLASS] PaintSample  [METHOD] <init> [RETURN_TYPE] Paint)   Paint paint [VARIABLES] Paint  paint  Dimension  preferredSize  boolean  
[P3_Replace_Literal]^this.preferredSize = new Dimension ( 83, 12 ) ;^75^^^^^73^76^this.preferredSize = new Dimension ( 80, 12 ) ;^[CLASS] PaintSample  [METHOD] <init> [RETURN_TYPE] Paint)   Paint paint [VARIABLES] Paint  paint  Dimension  preferredSize  boolean  
[P5_Replace_Variable]^return paint;^84^^^^^83^85^return this.paint;^[CLASS] PaintSample  [METHOD] getPaint [RETURN_TYPE] Paint   [VARIABLES] Paint  paint  Dimension  preferredSize  boolean  
[P8_Replace_Mix]^this.paint =  null;^93^^^^^92^95^this.paint = paint;^[CLASS] PaintSample  [METHOD] setPaint [RETURN_TYPE] void   Paint paint [VARIABLES] Paint  paint  Dimension  preferredSize  boolean  
[P7_Replace_Invocation]^getSize (  ) ;^94^^^^^92^95^repaint (  ) ;^[CLASS] PaintSample  [METHOD] setPaint [RETURN_TYPE] void   Paint paint [VARIABLES] Paint  paint  Dimension  preferredSize  boolean  
[P14_Delete_Statement]^^94^^^^^92^95^repaint (  ) ;^[CLASS] PaintSample  [METHOD] setPaint [RETURN_TYPE] void   Paint paint [VARIABLES] Paint  paint  Dimension  preferredSize  boolean  
[P5_Replace_Variable]^return preferredSize;^103^^^^^102^104^return this.preferredSize;^[CLASS] PaintSample  [METHOD] getPreferredSize [RETURN_TYPE] Dimension   [VARIABLES] Paint  paint  Dimension  preferredSize  boolean  
[P7_Replace_Invocation]^Dimension size = repaint (  ) ;^114^^^^^111^126^Dimension size = getSize (  ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P11_Insert_Donor_Statement]^Insets insets = getInsets (  ) ;Dimension size = getSize (  ) ;^114^^^^^111^126^Dimension size = getSize (  ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P14_Delete_Statement]^^114^^^^^111^126^Dimension size = getSize (  ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P11_Insert_Donor_Statement]^Dimension size = getSize (  ) ;Insets insets = getInsets (  ) ;^115^^^^^111^126^Insets insets = getInsets (  ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P14_Delete_Statement]^^115^116^^^^111^126^Insets insets = getInsets (  ) ; double xx = insets.left;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P1_Replace_Type]^int xx = insets.left;^116^^^^^111^126^double xx = insets.left;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^double xx = 2;^116^^^^^111^126^double xx = insets.left;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^double xx = insets.left.left;^116^^^^^111^126^double xx = insets.left;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^double xx = insets;^116^^^^^111^126^double xx = insets.left;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P11_Insert_Donor_Statement]^double yy = insets.top;double xx = insets.left;^116^^^^^111^126^double xx = insets.left;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P1_Replace_Type]^int yy = insets.top;^117^^^^^111^126^double yy = insets.top;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^double yy = insets.top.top;^117^^^^^111^126^double yy = insets.top;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^double yy = insets;^117^^^^^111^126^double yy = insets.top;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P11_Insert_Donor_Statement]^double xx = insets.left;double yy = insets.top;^117^^^^^111^126^double yy = insets.top;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P1_Replace_Type]^int ww = size.getWidth (  )  - insets.left - insets.right - 1;^118^^^^^111^126^double ww = size.getWidth (  )  - insets.left - insets.right - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P2_Replace_Operator]^double ww = size.getWidth (  )   !=  insets.left - insets.right - 1;^118^^^^^111^126^double ww = size.getWidth (  )  - insets.left - insets.right - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P2_Replace_Operator]^double ww = size.getWidth (  )   >  insets.left - insets.right - 1;^118^^^^^111^126^double ww = size.getWidth (  )  - insets.left - insets.right - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P3_Replace_Literal]^double ww = size.getWidth (  )  - insets.left - insets.right ;^118^^^^^111^126^double ww = size.getWidth (  )  - insets.left - insets.right - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^double ww = preferredSize.getWidth (  )  - insets.left - insets.right - 1;^118^^^^^111^126^double ww = size.getWidth (  )  - insets.left - insets.right - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^double ww = insets.right.getWidth (  )  - insets.left - size - 1;^118^^^^^111^126^double ww = size.getWidth (  )  - insets.left - insets.right - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^double ww = size.getWidth (  )  - insets.left.left - insets.right - 1;^118^^^^^111^126^double ww = size.getWidth (  )  - insets.left - insets.right - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^double ww = insets.left.getWidth (  )  - size - insets.right - 1;^118^^^^^111^126^double ww = size.getWidth (  )  - insets.left - insets.right - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^double ww = size.getWidth (  )  - insets.right - insets.left - 1;^118^^^^^111^126^double ww = size.getWidth (  )  - insets.left - insets.right - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P11_Insert_Donor_Statement]^double hh = size.getHeight (  )  - insets.top - insets.bottom - 1;double ww = size.getWidth (  )  - insets.left - insets.right - 1;^118^^^^^111^126^double ww = size.getWidth (  )  - insets.left - insets.right - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P14_Delete_Statement]^^118^119^^^^111^126^double ww = size.getWidth (  )  - insets.left - insets.right - 1; double hh = size.getHeight (  )  - insets.top - insets.bottom - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P1_Replace_Type]^int hh = size.getHeight (  )  - insets.top - insets.bottom - 1;^119^^^^^111^126^double hh = size.getHeight (  )  - insets.top - insets.bottom - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P2_Replace_Operator]^double hh = size.getHeight (  )   ||  insets.top - insets.bottom - 1;^119^^^^^111^126^double hh = size.getHeight (  )  - insets.top - insets.bottom - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P2_Replace_Operator]^double hh = size.getHeight (  )   >>  insets.top - insets.bottom - 1;^119^^^^^111^126^double hh = size.getHeight (  )  - insets.top - insets.bottom - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P2_Replace_Operator]^double hh = size.getHeight (  )   &  insets.top - insets.bottom - 1;^119^^^^^111^126^double hh = size.getHeight (  )  - insets.top - insets.bottom - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P3_Replace_Literal]^double hh = size.getHeight (  )  - insets.top - insets.bottom - ;^119^^^^^111^126^double hh = size.getHeight (  )  - insets.top - insets.bottom - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^double hh = preferredSize.getHeight (  )  - insets.top - insets.bottom - 1;^119^^^^^111^126^double hh = size.getHeight (  )  - insets.top - insets.bottom - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^double hh = insets.top.getHeight (  )  - size - insets.bottom - 1;^119^^^^^111^126^double hh = size.getHeight (  )  - insets.top - insets.bottom - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^double hh = insets.getHeight (  )  - size.top - insets.bottom - 1;^119^^^^^111^126^double hh = size.getHeight (  )  - insets.top - insets.bottom - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^double hh = insets.bottom.getHeight (  )  - insets.top - size - 1;^119^^^^^111^126^double hh = size.getHeight (  )  - insets.top - insets.bottom - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P11_Insert_Donor_Statement]^double ww = size.getWidth (  )  - insets.left - insets.right - 1;double hh = size.getHeight (  )  - insets.top - insets.bottom - 1;^119^^^^^111^126^double hh = size.getHeight (  )  - insets.top - insets.bottom - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P14_Delete_Statement]^^119^^^^^111^126^double hh = size.getHeight (  )  - insets.top - insets.bottom - 1;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P4_Replace_Constructor]^Rectangle2D area = new Rectangle2D.Double (  yy, ww, hh ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P4_Replace_Constructor]^Rectangle2D area = new Rectangle2D.Double ( xx,  ww, hh ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P4_Replace_Constructor]^Rectangle2D area = new Rectangle2D.Double ( xx, yy,  hh ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P4_Replace_Constructor]^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^Rectangle2D area = new Rectangle2D.Double ( ww, yy, ww, hh ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^Rectangle2D area = new Rectangle2D.Double ( xx, yy, yy, hh ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, xx ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^Rectangle2D area = new Rectangle2D.Double ( hh, yy, ww, xx ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^Rectangle2D area = new Rectangle2D.Double ( xx, ww, yy, hh ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^Rectangle2D area = new Rectangle2D.Double ( ww, yy, xx, hh ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^Rectangle2D area = new Rectangle2D.Double ( xx, yy, hh, ww ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P8_Replace_Mix]^Rectangle2D area = new Rectangle2D.Double ( xx, yy, xx, hh ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^Rectangle2D area = new Rectangle2D.Double ( yy, yy, ww, hh ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^Rectangle2D area = new Rectangle2D.Double ( xx, xx, ww, hh ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, yy ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^Rectangle2D area = new Rectangle2D.Double ( xx, hh, ww, yy ) ;^120^^^^^111^126^Rectangle2D area = new Rectangle2D.Double ( xx, yy, ww, hh ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P5_Replace_Variable]^g2.setPaint ( paint ) ;^121^^^^^111^126^g2.setPaint ( this.paint ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P7_Replace_Invocation]^g2.fill ( this.paint ) ;^121^^^^^111^126^g2.setPaint ( this.paint ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P14_Delete_Statement]^^121^122^^^^111^126^g2.setPaint ( this.paint ) ; g2.fill ( area ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P11_Insert_Donor_Statement]^g2.setPaint ( Color.black ) ;g2.setPaint ( this.paint ) ;^121^^^^^111^126^g2.setPaint ( this.paint ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P8_Replace_Mix]^g2 .setPaint ( paint )  ;^122^^^^^111^126^g2.fill ( area ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P14_Delete_Statement]^^122^^^^^111^126^g2.fill ( area ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P11_Insert_Donor_Statement]^g2.draw ( area ) ;g2.fill ( area ) ;^122^^^^^111^126^g2.fill ( area ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P7_Replace_Invocation]^g2.fill ( Color.black ) ;^123^^^^^111^126^g2.setPaint ( Color.black ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P8_Replace_Mix]^g2.setPaint ( Color.null ) ;^123^^^^^111^126^g2.setPaint ( Color.black ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P14_Delete_Statement]^^123^124^^^^111^126^g2.setPaint ( Color.black ) ; g2.draw ( area ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P11_Insert_Donor_Statement]^g2.setPaint ( this.paint ) ;g2.setPaint ( Color.black ) ;^123^^^^^111^126^g2.setPaint ( Color.black ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P14_Delete_Statement]^^124^^^^^111^126^g2.draw ( area ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
[P11_Insert_Donor_Statement]^g2.fill ( area ) ;g2.draw ( area ) ;^124^^^^^111^126^g2.draw ( area ) ;^[CLASS] PaintSample  [METHOD] paintComponent [RETURN_TYPE] void   Graphics g [VARIABLES] Graphics  g  Insets  insets  boolean  double  hh  ww  xx  yy  Rectangle2D  area  Paint  paint  Dimension  preferredSize  size  Graphics2D  g2  
